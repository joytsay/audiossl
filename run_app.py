import argparse
import html
import inspect
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from audiossl.methods.atstframe.downstream.Inference_audioset_strong import (
    InferenceAudioSetStrong,
    infer_label_path,
    load_label_display_names,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RTSP_URL = "rtsp://admin:Admin123@192.168.5.157:554/profile1"
DEFAULT_LABEL_TSV = REPO_ROOT / "mid_office_indoor_display_name.tsv"
# DEFAULT_CKPT = REPO_ROOT / "models" / "atst_ft_StrongAS_eps28.ckpt"
DEFAULT_CKPT = REPO_ROOT / "logs" /"as_strong_small" / "frameatst_small_freeze"  / "last.ckpt"
LABEL_TSV_CHOICES = [
    "mid_10_display_name.tsv",
    "mid_20_display_name.tsv",
    "mid_indoor_cctv_display_name.tsv",
    "mid_industrial_hazard_display_name.tsv",
    "mid_human_annoying_display_name.tsv",
    "mid_office_indoor_display_name.tsv",
    "mid_street_surveillance_display_name.tsv",
    "mid_to_display_name.tsv",
]
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
DEFAULT_SILENCE_GATE_ENABLED = True
DEFAULT_SILENCE_RMS_THRESHOLD = 0.005
DEFAULT_SILENCE_PEAK_THRESHOLD = 0.03
DEFAULT_TOP1_ALERT_THRESHOLD = 0.01
FFMPEG_IGNORED_ERROR_PATTERNS = (
    r"\[s16le @ [^\]]+\] Application provided invalid, non monotonically increasing dts to muxer in stream \d+: \d+ >= \d+",
    r"error parsing debug value debug=0",
    r"Enter command: <target>\|all <time>\|-1 <command>\[ <argument>\]",
)


@dataclass
class PredictionState:
    connected: bool = False
    running: bool = False
    rtsp_url: str = DEFAULT_RTSP_URL
    label_tsv: str = str(DEFAULT_LABEL_TSV)
    ckpt_path: str = str(DEFAULT_CKPT)
    device: str = "cuda"
    status: str = "Idle"
    error: str = ""
    frame_index: int = 0
    updated_at: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    silence_gate_enabled: bool = DEFAULT_SILENCE_GATE_ENABLED
    silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD
    silence_peak_threshold: float = DEFAULT_SILENCE_PEAK_THRESHOLD
    top_labels: List[Dict[str, Any]] = field(default_factory=list)
    active_labels: List[str] = field(default_factory=list)


class LabelMapper:
    def __init__(self, label_tsv: Path, model_labels: Optional[Sequence[str]] = None):
        self.common_mids = list(model_labels) if model_labels is not None else self._read_common_mids()
        self.mid_to_name = self._read_mid_names()
        self.selected = self._read_selected(label_tsv)

    def _read_common_mids(self) -> List[str]:
        path = REPO_ROOT / "common_labels.txt"
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _read_mid_names(self) -> Dict[str, str]:
        path = REPO_ROOT / "mid_to_display_name.tsv"
        names = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    names[parts[0]] = parts[1]
        return names

    def _read_selected(self, label_tsv: Path) -> List[int]:
        wanted = set()
        with label_tsv.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if parts and parts[0]:
                    wanted.add(parts[0])
        indices = [i for i, mid in enumerate(
            self.common_mids) if mid in wanted]
        if not indices:
            indices = list(range(len(self.common_mids)))
        return indices

    def topk(self, scores: Sequence[float], k: int = 3) -> List[Dict[str, Any]]:
        selected_scores = [(idx, float(scores[idx])) for idx in self.selected if idx < len(scores)]
        selected_scores.sort(key=lambda item: item[1], reverse=True)
        results = []
        for rank, (label_index, confidence) in enumerate(selected_scores[:k], start=1):
            mid = self.common_mids[label_index]
            results.append(
                {
                    "rank": rank,
                    "mid": mid,
                    "label": self.mid_to_name.get(mid, mid),
                    "confidence": confidence,
                }
            )
        return results


class RtspInferenceWorker:
    def __init__(self):
        self.state = PredictionState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    def snapshot(self) -> PredictionState:
        with self._lock:
            return PredictionState(
                connected=self.state.connected,
                running=self.state.running,
                rtsp_url=self.state.rtsp_url,
                label_tsv=self.state.label_tsv,
                ckpt_path=self.state.ckpt_path,
                device=self.state.device,
                status=self.state.status,
                error=self.state.error,
                frame_index=self.state.frame_index,
                updated_at=self.state.updated_at,
                rms=self.state.rms,
                peak=self.state.peak,
                silence_gate_enabled=self.state.silence_gate_enabled,
                silence_rms_threshold=self.state.silence_rms_threshold,
                silence_peak_threshold=self.state.silence_peak_threshold,
                top_labels=list(self.state.top_labels),
                active_labels=list(self.state.active_labels),
            )

    def start(
        self,
        rtsp_url: str,
        label_tsv: str,
        ckpt_path: str,
        device: str = "cuda",
        window_seconds: float = 10.0,
        update_every_frames: int = 5,
        top_n: int = 3,
        silence_gate_enabled: bool = DEFAULT_SILENCE_GATE_ENABLED,
        silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
        silence_peak_threshold: float = DEFAULT_SILENCE_PEAK_THRESHOLD,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        with self._lock:
            self.state = PredictionState(
                connected=True,
                running=True,
                rtsp_url=rtsp_url,
                label_tsv=label_tsv,
                ckpt_path=ckpt_path,
                device=device,
                status="Starting",
                error="",
                silence_gate_enabled=silence_gate_enabled,
                silence_rms_threshold=silence_rms_threshold,
                silence_peak_threshold=silence_peak_threshold,
            )
        self._thread = threading.Thread(
            target=self._run,
            args=(rtsp_url, label_tsv, ckpt_path, device,
                  window_seconds, update_every_frames, top_n,
                  silence_gate_enabled, silence_rms_threshold,
                  silence_peak_threshold),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            self.state.connected = False
            self.state.running = False
            if self.state.status not in {"Idle", "Error"}:
                self.state.status = "Stopped"

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self.state.status = status
            self.state.error = error
            self.state.running = status not in {"Error", "Stopped"}

    def _set_quiet(self, frame_index: int, rms: float, peak: float) -> None:
        with self._lock:
            self.state.status = "Quiet"
            self.state.running = True
            self.state.frame_index = frame_index
            self.state.updated_at = time.time()
            self.state.rms = rms
            self.state.peak = peak
            self.state.top_labels = []

    def _set_prediction(
        self,
        frame_index: int,
        top_labels: List[Dict[str, Any]],
        rms: float,
        peak: float,
    ) -> None:
        with self._lock:
            self.state.status = "Streaming"
            self.state.running = True
            self.state.frame_index = frame_index
            self.state.updated_at = time.time()
            self.state.rms = rms
            self.state.peak = peak
            self.state.top_labels = top_labels

    def _set_active_labels(self, labels: Sequence[str]) -> None:
        with self._lock:
            self.state.active_labels = list(labels)

    def _run(
        self,
        rtsp_url: str,
        label_tsv: str,
        ckpt_path: str,
        device_name: str,
        window_seconds: float,
        update_every_frames: int,
        top_n: int,
        silence_gate_enabled: bool,
        silence_rms_threshold: float,
        silence_peak_threshold: float,
    ) -> None:
        try:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "ffmpeg is required but was not found on PATH")

            device = _resolve_device(torch, device_name)
            model = InferenceAudioSetStrong(str(Path(ckpt_path).expanduser()))
            self._set_active_labels(model.display_labels)
            labels = LabelMapper(Path(label_tsv).expanduser(), model.display_labels)
            model.to(device)
            if hasattr(model, "transform"):
                model.transform = _move_transform_to_device(
                    model.transform, device)
            model.eval()

            samples_per_read = max(TARGET_SAMPLE_RATE // 2, 1)
            bytes_per_read = samples_per_read * CHANNELS * BYTES_PER_SAMPLE
            max_samples = int(TARGET_SAMPLE_RATE * window_seconds)
            audio = np.zeros(0, dtype=np.float32)
            frame_index = 0

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "warning",
                "-fflags",
                "+genpts+discardcorrupt",
                "-use_wallclock_as_timestamps",
                "1",
                "-rtsp_transport",
                "tcp",
                "-i",
                rtsp_url,
                "-vn",
                "-ac",
                str(CHANNELS),
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-f",
                "s16le",
                "-",
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            stderr_queue = _drain_stderr(self._process)
            self._set_status("Buffering")

            while not self._stop_event.is_set():
                if self._process.stdout is None:
                    raise RuntimeError("ffmpeg stdout is unavailable")
                chunk = self._process.stdout.read(bytes_per_read)
                if not chunk:
                    err = _collect_stderr(stderr_queue)
                    raise RuntimeError(_format_stream_error(err))

                pcm = np.frombuffer(chunk, dtype=np.int16).astype(
                    np.float32) / 32768.0
                audio = np.concatenate([audio, pcm])
                if audio.size > max_samples:
                    audio = audio[-max_samples:]
                if audio.size < max_samples:
                    continue

                rms = float(np.sqrt(np.mean(audio ** 2)))
                peak = float(np.max(np.abs(audio)))
                frame_index += update_every_frames
                if (
                    silence_gate_enabled
                    and rms < silence_rms_threshold
                    and peak < silence_peak_threshold
                ):
                    self._set_quiet(frame_index, rms, peak)
                    continue

                wav = torch.from_numpy(audio.copy()).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model.predict(wav)
                scores = prediction[0, :, -update_every_frames:].mean(dim=1)
                top_labels = labels.topk(scores.detach().cpu().tolist(), k=top_n)
                self._set_prediction(frame_index, top_labels, rms, peak)

        except Exception as exc:
            self._set_status("Error", str(exc))
        finally:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
            self._process = None
            with self._lock:
                self.state.connected = False
                if self.state.status != "Error":
                    self.state.running = False


def _resolve_device(torch: Any, device_name: str):
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _move_transform_to_device(transform: Any, device: Any) -> Any:
    if hasattr(transform, "transforms"):
        for item in transform.transforms:
            if hasattr(item, "to"):
                item.to(device)
    elif hasattr(transform, "to"):
        transform.to(device)
    return transform


def _drain_stderr(process: subprocess.Popen) -> "queue.Queue[str]":
    stderr_queue: "queue.Queue[str]" = queue.Queue()

    def drain() -> None:
        if process.stderr is None:
            return
        for raw in iter(process.stderr.readline, b""):
            if raw:
                stderr_queue.put(raw.decode("utf-8", errors="replace").strip())

    threading.Thread(target=drain, daemon=True).start()
    return stderr_queue


def _collect_stderr(stderr_queue: "queue.Queue[str]") -> str:
    lines = []
    while True:
        try:
            line = stderr_queue.get_nowait()
        except queue.Empty:
            break
        if line:
            lines.append(line)
    return "\n".join(lines[-8:])


def _format_stream_error(stderr: str) -> str:
    filtered = stderr
    for pattern in FFMPEG_IGNORED_ERROR_PATTERNS:
        filtered = re.sub(pattern, "", filtered)
    filtered = "\n".join(
        line.strip() for line in filtered.splitlines() if line.strip()
    )
    if filtered:
        return filtered
    return "RTSP stream ended or ffmpeg stopped without a fatal error message"


worker = RtspInferenceWorker()


def create_fastapi_app():
    class ConnectRequest(BaseModel):
        rtsp_url: str = DEFAULT_RTSP_URL
        label_tsv: str = str(DEFAULT_LABEL_TSV)
        ckpt_path: str = str(DEFAULT_CKPT)
        device: str = "cuda"
        window_seconds: float = 10.0
        update_every_frames: int = 5
        top_n: int = 3
        silence_gate_enabled: bool = DEFAULT_SILENCE_GATE_ENABLED
        silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD
        silence_peak_threshold: float = DEFAULT_SILENCE_PEAK_THRESHOLD

    app = FastAPI(title="AudioSSL RTSP ATST-Frame")

    @app.get("/status")
    def status():
        return worker.snapshot().__dict__

    @app.get("/results")
    def results():
        snapshot = worker.snapshot()
        return {
            "status": snapshot.status,
            "error": snapshot.error,
            "frame_index": snapshot.frame_index,
            "updated_at": snapshot.updated_at,
            "rms": snapshot.rms,
            "peak": snapshot.peak,
            "silence_gate_enabled": snapshot.silence_gate_enabled,
            "silence_rms_threshold": snapshot.silence_rms_threshold,
            "silence_peak_threshold": snapshot.silence_peak_threshold,
            "top_labels": snapshot.top_labels,
        }

    @app.post("/connect")
    def connect(request: ConnectRequest):
        worker.start(
            request.rtsp_url,
            request.label_tsv,
            request.ckpt_path,
            request.device,
            request.window_seconds,
            request.update_every_frames,
            request.top_n,
            request.silence_gate_enabled,
            request.silence_rms_threshold,
            request.silence_peak_threshold,
        )
        return worker.snapshot().__dict__

    @app.post("/disconnect")
    def disconnect():
        worker.stop()
        return worker.snapshot().__dict__

    return app


def create_gradio_app():
    def connect(
        rtsp_url,
        label_tsv,
        ckpt_path,
        device,
        window_seconds,
        update_every_frames,
        top_n,
        top1_alert_threshold,
        silence_gate_enabled,
        silence_rms_threshold,
        silence_peak_threshold,
    ):
        worker.start(rtsp_url, label_tsv, ckpt_path, device,
                     window_seconds, update_every_frames, top_n,
                     silence_gate_enabled, silence_rms_threshold,
                     silence_peak_threshold)
        snapshot = worker.snapshot()
        snapshot.active_labels = load_checkpoint_labels(ckpt_path)
        return _format_outputs(snapshot, top1_alert_threshold)

    def disconnect(top1_alert_threshold):
        worker.stop()
        return _format_outputs(worker.snapshot(), top1_alert_threshold)

    def poll(top1_alert_threshold):
        return _format_outputs(worker.snapshot(), top1_alert_threshold)

    def select_label_tsv(filename):
        if not filename:
            path = str(DEFAULT_LABEL_TSV)
        else:
            path = str(REPO_ROOT / filename)
        snapshot = worker.snapshot()
        snapshot.label_tsv = path
        return path, _format_active_labels(snapshot)

    with gr.Blocks(title="GeoVision Sound-Event-Detection") as demo:
        gr.Markdown("# GeoVision Sound-Event-Detection")
        with gr.Row():
            rtsp_url = gr.Textbox(
                label="RTSP URL", value=DEFAULT_RTSP_URL, scale=3)
            device = gr.Dropdown(label="Device", choices=[
                                 "cpu", "cuda"], value="cuda")
        with gr.Row():
            label_choice = gr.Dropdown(
                label="Label TSV preset",
                choices=LABEL_TSV_CHOICES,
                value=DEFAULT_LABEL_TSV.name,
                scale=1,
            )
            label_tsv = gr.Textbox(
                label="Label TSV", value=str(DEFAULT_LABEL_TSV), scale=2)
            ckpt_path = gr.Textbox(
                label="Checkpoint", value=str(DEFAULT_CKPT), scale=2)
        with gr.Row():
            window_seconds = gr.Slider(
                label="Audio window seconds",
                minimum=2,
                maximum=20,
                value=10,
                step=1,
            )
            update_every_frames = gr.Slider(
                label="Average latest N model frames",
                minimum=5,
                maximum=50,
                value=5,
                step=5,
            )
            top_n = gr.Slider(
                label="Top N labels",
                minimum=1,
                maximum=20,
                value=3,
                step=1,
            )
            top1_alert_threshold = gr.Slider(
                label="Top 1 alert confidence",
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_TOP1_ALERT_THRESHOLD,
                step=0.005,
            )
        with gr.Row():
            silence_gate_enabled = gr.Checkbox(
                label="Silence gate",
                value=DEFAULT_SILENCE_GATE_ENABLED,
            )
            silence_rms_threshold = gr.Slider(
                label="RMS threshold",
                minimum=0.0,
                maximum=0.1,
                value=DEFAULT_SILENCE_RMS_THRESHOLD,
                step=0.001,
            )
            silence_peak_threshold = gr.Slider(
                label="Peak threshold",
                minimum=0.0,
                maximum=0.5,
                value=DEFAULT_SILENCE_PEAK_THRESHOLD,
                step=0.005,
            )
        with gr.Row():
            connect_btn = gr.Button("Connect", variant="primary")
            disconnect_btn = gr.Button("Disconnect")
        top1_alert = gr.HTML(label="Top 1 alert")
        output = gr.JSON(label="Live top labels")
        all_labels = gr.JSON(label="All labels",
                             value=_format_active_labels(PredictionState(
                                 label_tsv=str(DEFAULT_LABEL_TSV),
                                 active_labels=load_checkpoint_labels(str(DEFAULT_CKPT)),
                             )))
        timer = gr.Timer(value=1.0, active=True)

        connect_btn.click(
            connect,
            inputs=[rtsp_url, label_tsv, ckpt_path, device,
                    window_seconds, update_every_frames, top_n,
                    top1_alert_threshold,
                    silence_gate_enabled, silence_rms_threshold,
                    silence_peak_threshold],
            outputs=[output, top1_alert, all_labels],
        )
        disconnect_btn.click(
            disconnect,
            inputs=top1_alert_threshold,
            outputs=[output, top1_alert, all_labels],
        )
        label_choice.change(select_label_tsv, inputs=label_choice, outputs=[
                            label_tsv, all_labels])
        timer.tick(poll, inputs=top1_alert_threshold, outputs=[output, top1_alert, all_labels])

    return demo


def create_gradio_theme():
    return gr.themes.Base()


def _format_outputs(snapshot: PredictionState, top1_alert_threshold: float):
    return (
        _format_snapshot(snapshot),
        _format_top1_alert(snapshot, top1_alert_threshold),
        _format_active_labels(snapshot),
    )


def _format_top1_alert(snapshot: PredictionState, top1_alert_threshold: float) -> str:
    if not snapshot.top_labels:
        return ""

    top_label = snapshot.top_labels[0]
    confidence = float(top_label.get("confidence", 0.0))
    if confidence < float(top1_alert_threshold):
        return ""

    label = html.escape(str(top_label.get("label", "Unknown")))
    return (
        '<button style="'
        'background:#ffeb00;'
        'color:#c00000;'
        'border:3px solid #c00000;'
        'border-radius:6px;'
        'font-size:28px;'
        'font-weight:800;'
        'padding:16px 24px;'
        'width:100%;'
        'text-align:center;'
        '">'
        f'ALERT: {label} ({confidence:.3f})'
        '</button>'
    )



def _format_snapshot(snapshot: PredictionState) -> Dict[str, Any]:
    return {
        "status": snapshot.status,
        "error": snapshot.error,
        "rtsp_url": snapshot.rtsp_url,
        "label_tsv": snapshot.label_tsv,
        "ckpt_path": snapshot.ckpt_path,
        "frame_index": snapshot.frame_index,
        "updated_at": snapshot.updated_at,
        "rms": snapshot.rms,
        "peak": snapshot.peak,
        "silence_gate_enabled": snapshot.silence_gate_enabled,
        "silence_rms_threshold": snapshot.silence_rms_threshold,
        "silence_peak_threshold": snapshot.silence_peak_threshold,
        "top_labels": snapshot.top_labels,
        "active_labels": snapshot.active_labels,
    }


def _format_active_labels(snapshot: PredictionState) -> Dict[str, Any]:
    if snapshot.active_labels:
        return {
            "source": "active checkpoint labels",
            "count": len(snapshot.active_labels),
            "labels": snapshot.active_labels,
        }
    return load_tsv_labels(snapshot.label_tsv)


def load_checkpoint_labels(ckpt_path: str) -> List[str]:
    try:
        checkpoint = torch.load(str(Path(ckpt_path).expanduser()), map_location="cpu")
        state_dict = checkpoint["state_dict"]
        num_labels = state_dict["head.linear.weight"].shape[0]
        label_path = infer_label_path(num_labels)
        if label_path:
            labels = load_label_display_names(label_path)
        else:
            labels = [str(i) for i in range(num_labels)]
        if len(labels) < num_labels:
            labels.extend(str(i) for i in range(len(labels), num_labels))
        return labels
    except Exception:
        return []


def load_tsv_labels(label_tsv: str) -> Dict[str, Any]:
    path = Path(label_tsv).expanduser()
    if not path.exists():
        return {"file_name": path.name, "count": 0, "labels": [], "error": f"{path} does not exist"}

    mid_to_name = {}
    names_path = REPO_ROOT / "mid_to_display_name.tsv"
    if names_path.exists():
        with names_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    mid_to_name[parts[0]] = parts[1]

    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if not parts or not parts[0]:
                continue
            mid = parts[0]
            label = parts[1] if len(
                parts) == 2 and parts[1] else mid_to_name.get(mid, mid)
            labels.append(label)
    return {"file_name": path.name, "count": len(labels), "labels": labels}


def create_app():
    app = create_fastapi_app()
    demo = create_gradio_app()
    try:
        mount_kwargs = {
            "theme": gr.themes.Base(),
        }
        supported = inspect.signature(gr.mount_gradio_app).parameters
        mount_kwargs = {
            key: value for key, value in mount_kwargs.items() if key in supported
        }
        app = gr.mount_gradio_app(app, demo, path="/", **mount_kwargs)
    except AttributeError as exc:
        raise RuntimeError(
            "This app needs a Gradio version with mount_gradio_app") from exc
    return app


class LazyApp:
    def __init__(self):
        self._app = None
        self._lock = threading.Lock()

    def _get_app(self):
        if self._app is None:
            with self._lock:
                if self._app is None:
                    self._app = create_app()
        return self._app

    async def __call__(self, scope, receive, send):
        app = self._get_app()
        await app(scope, receive, send)


app = LazyApp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
