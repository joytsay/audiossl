import argparse
import inspect
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RTSP_URL = "rtsp://admin:Admin123@192.168.5.157:554/profile1"
DEFAULT_LABEL_TSV = REPO_ROOT / "mid_street_surveillance_display_name.tsv"
DEFAULT_CKPT = REPO_ROOT / "models" / "atst_ft_StrongAS_eps28.ckpt"
LABEL_TSV_CHOICES = [
    "mid_10_display_name.tsv",
    "mid_20_display_name.tsv",
    "mid_indoor_cctv_display_name.tsv",
    "mid_industrial_hazard_display_name.tsv",
    "mid_street_surveillance_display_name.tsv",
    "mid_to_display_name.tsv",
]
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2


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
    top_labels: List[Dict[str, Any]] = field(default_factory=list)


class LabelMapper:
    def __init__(self, label_tsv: Path):
        self.common_mids = self._read_common_mids()
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
            raise ValueError(f"No labels from {label_tsv} match common_labels.txt")
        return indices

    def topk(self, scores: Sequence[float], k: int = 3) -> List[Dict[str, Any]]:
        selected_scores = [(idx, float(scores[idx])) for idx in self.selected]
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
                top_labels=list(self.state.top_labels),
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
            )
        self._thread = threading.Thread(
            target=self._run,
            args=(rtsp_url, label_tsv, ckpt_path, device,
                  window_seconds, update_every_frames, top_n),
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

    def _set_prediction(self, frame_index: int, top_labels: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.state.status = "Streaming"
            self.state.running = True
            self.state.frame_index = frame_index
            self.state.updated_at = time.time()
            self.state.top_labels = top_labels

    def _run(
        self,
        rtsp_url: str,
        label_tsv: str,
        ckpt_path: str,
        device_name: str,
        window_seconds: float,
        update_every_frames: int,
        top_n: int,
    ) -> None:
        try:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "ffmpeg is required but was not found on PATH")

            np = _import_numpy()
            torch = _import_torch()
            InferenceAudioSetStrong = _import_inference_model()

            device = _resolve_device(torch, device_name)
            labels = LabelMapper(Path(label_tsv).expanduser())
            model = InferenceAudioSetStrong(str(Path(ckpt_path).expanduser()))
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
                "-loglevel",
                "warning",
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
                    raise RuntimeError(err or "RTSP stream ended")

                pcm = np.frombuffer(chunk, dtype=np.int16).astype(
                    np.float32) / 32768.0
                audio = np.concatenate([audio, pcm])
                if audio.size > max_samples:
                    audio = audio[-max_samples:]
                if audio.size < max_samples:
                    continue

                wav = torch.from_numpy(audio.copy()).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model.predict(wav)
                scores = prediction[0, :, -update_every_frames:].mean(dim=1)
                top_labels = labels.topk(scores.detach().cpu().tolist(), k=top_n)
                frame_index += update_every_frames
                self._set_prediction(frame_index, top_labels)

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


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install numpy to decode the ffmpeg PCM stream") from exc
    return np


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install torch and torchaudio to run inference") from exc
    return torch


def _import_inference_model():
    try:
        from audiossl.methods.atstframe.downstream.Inference_audioset_strong import (
            InferenceAudioSetStrong,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import ATST-Frame inference model: {exc}") from exc
    return InferenceAudioSetStrong


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


worker = RtspInferenceWorker()


def create_fastapi_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "Install fastapi and pydantic to run the web API") from exc

    class ConnectRequest(BaseModel):
        rtsp_url: str = DEFAULT_RTSP_URL
        label_tsv: str = str(DEFAULT_LABEL_TSV)
        ckpt_path: str = str(DEFAULT_CKPT)
        device: str = "cuda"
        window_seconds: float = 10.0
        update_every_frames: int = 5
        top_n: int = 3

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
        )
        return worker.snapshot().__dict__

    @app.post("/disconnect")
    def disconnect():
        worker.stop()
        return worker.snapshot().__dict__

    return app


def create_gradio_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to run the UI") from exc

    def connect(rtsp_url, label_tsv, ckpt_path, device, window_seconds, update_every_frames, top_n):
        worker.start(rtsp_url, label_tsv, ckpt_path, device,
                     window_seconds, update_every_frames, top_n)
        return _format_snapshot(worker.snapshot())

    def disconnect():
        worker.stop()
        return _format_snapshot(worker.snapshot())

    def poll():
        return _format_snapshot(worker.snapshot())

    def select_label_tsv(filename):
        if not filename:
            path = str(DEFAULT_LABEL_TSV)
        else:
            path = str(REPO_ROOT / filename)
        return path, load_tsv_labels(path)

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
        with gr.Row():
            connect_btn = gr.Button("Connect", variant="primary")
            disconnect_btn = gr.Button("Disconnect")
        output = gr.JSON(label="Live top labels")
        all_labels = gr.JSON(label="All labels",
                             value=load_tsv_labels(str(DEFAULT_LABEL_TSV)))
        timer = gr.Timer(value=1.0, active=True)

        connect_btn.click(
            connect,
            inputs=[rtsp_url, label_tsv, ckpt_path, device,
                    window_seconds, update_every_frames, top_n],
            outputs=output,
        )
        disconnect_btn.click(disconnect, outputs=output)
        label_choice.change(select_label_tsv, inputs=label_choice, outputs=[
                            label_tsv, all_labels])
        timer.tick(poll, outputs=output)

    return demo


def create_gradio_theme():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to run the UI") from exc

    return gr.themes.Base()



def _format_snapshot(snapshot: PredictionState) -> Dict[str, Any]:
    return {
        "status": snapshot.status,
        "error": snapshot.error,
        "rtsp_url": snapshot.rtsp_url,
        "label_tsv": snapshot.label_tsv,
        "ckpt_path": snapshot.ckpt_path,
        "frame_index": snapshot.frame_index,
        "updated_at": snapshot.updated_at,
        "top_labels": snapshot.top_labels,
    }


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
        import gradio as gr

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
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install uvicorn to run the server") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
