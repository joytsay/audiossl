
import argparse
import os
import struct
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from audiossl.methods.atstframe.downstream.comparison_models.models.frame_atst import FrameAST_base
from audiossl.methods.atstframe.downstream.utils_as_strong.model_as_strong import LinearHead
from audiossl.transforms.common import MinMax


REPO_ROOT = Path(__file__).resolve().parents[4]


def load_label_display_names():
    common_labels_path = REPO_ROOT / "common_labels.txt"
    mid_to_display_name_path = REPO_ROOT / "mid_to_display_name.tsv"

    with common_labels_path.open() as f:
        label_mids = [line.strip() for line in f if line.strip()]

    mid_to_display_name = {}
    with mid_to_display_name_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                mid_to_display_name[parts[0]] = parts[1]

    return [mid_to_display_name.get(mid, mid) for mid in label_mids]


DISPLAY_LABELS = load_label_display_names()
MEL_HOP_SAMPLES = 160
TARGET_SAMPLE_RATE = 16000
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_ALAW = 0x0006
WAVE_FORMAT_MULAW = 0x0007
WAVE_FORMAT_VMS_G726 = 0x4148


@dataclass
class AudioMetadata:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str
    codec_tag: Optional[str] = None


class InferenceAudioSetStrong(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.encoder = FrameAST_base()
        self.head = LinearHead(768, 407, use_norm=False, affine=False)
        self._load_ckpt(ckpt_path)
        self.transform = self._transform()
        self.seconds_per_prediction_frame = (
            self.encoder.patch_w * MEL_HOP_SAMPLES / TARGET_SAMPLE_RATE
        )

    def _transform(self):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        normalize = MinMax(min=-79.6482, max=50.6842)
        return transforms.Compose([melspec_t,
                                   to_db,
                                   normalize])

    def _load_ckpt(self, ckpt_path):
        s = torch.load(ckpt_path, map_location="cpu")
        state_dict = s["state_dict"]
        replaced_state_dict = {}
        for key in state_dict.keys():
            replaced_state_dict[key.replace(
                "encoder.encoder", "encoder")] = state_dict[key]

        self.load_state_dict(replaced_state_dict)

    def _prepare_wav(self, wav):
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(1)
        else:
            assert len(wav.shape) == 3
        return wav

    def _chunk_mel(self, mel):
        chunk_len = 1001  # 10 secnods, consistent with the length of positional embedding
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        for i in range(num_chunks):
            start = i*chunk_len
            end = min((i+1) * chunk_len, total_len)
            if end > start:
                yield i, mel[:, :, :, start:end]

    def predict(self, wav):
        """
        ==================================================
        args:
        wav: torch.tensor in the shape of [1,N] or [B,1,N] 
        """"""
        return:
             retured prediction in the shape of [1,407,T] or [B,407,T]
        """
        wav = self._prepare_wav(wav)
        mel = self.transform(wav)
        output = []
        for _, mel_chunk in self._chunk_mel(mel):
            len_chunk = torch.tensor(
                [mel_chunk.shape[-1]]).expand(mel.shape[0]).to(wav.device)
            output_chunk = self.encoder.get_intermediate_layers(
                mel_chunk, len_chunk, n=1, scene=False)
            output.append(output_chunk)
        output = torch.cat(output, dim=1)
        output = self.head(output)
        return output

    def get_attention(self, wav):
        wav = self._prepare_wav(wav)
        mel = self.transform(wav)
        attentions = []
        mel_chunks = []
        for _, mel_chunk in self._chunk_mel(mel):
            attentions.append(
                self.encoder.get_last_selfattention(mel_chunk)[-1])
            mel_chunks.append(mel_chunk)
        return mel, mel_chunks, attentions


def plot_spec(x, save_path):
    plt.figure()
    plt.axis("off")
    plt.pcolormesh(x)
    plt.xticks([])
    plt.yticks([])
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=400, pad_inches=-0.01, transparent=True)
    plt.close()


def set_animated_y_labels(ax, base_labels, highlight_positions, label_scores):
    cmap = plt.get_cmap("viridis")
    anchor_values = [1.0, 0.6, 0.0]
    updated_labels = list(base_labels)

    for label_idx in highlight_positions:
        if label_idx < len(updated_labels) and label_idx < len(label_scores):
            updated_labels[label_idx] = f"{base_labels[label_idx]}  {float(label_scores[label_idx]):.3f}"

    ax.set_yticklabels(updated_labels)
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_bbox(None)
        label.set_fontweight("normal")
        label.set_color("black")

    for label_idx, score in zip(highlight_positions, anchor_values):
        if label_idx >= len(labels):
            continue
        label = labels[label_idx]
        color = cmap(float(score))
        label.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=None))
        label.set_fontweight("bold")
        label.set_color("red")


def plot_prediction(prediction, save_path, top_k=10, use_sec=False, seconds_per_frame=None):
    frame_scores = prediction[0].detach().cpu()
    mean_scores = frame_scores.mean(dim=1)
    top_k = min(top_k, frame_scores.shape[0])
    top_indices = torch.topk(mean_scores, k=top_k).indices

    plt.figure(figsize=(12, 4))
    if use_sec:
        assert seconds_per_frame is not None
        num_frames = frame_scores.shape[1]
        duration_sec = num_frames * seconds_per_frame
        plt.imshow(
            frame_scores[top_indices].numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, duration_sec, 0, top_k],
        )
        plt.xlabel("Seconds")
        plt.yticks(torch.arange(top_k).float() + 0.5,
                   [DISPLAY_LABELS[i] for i in top_indices.tolist()])
    else:
        plt.imshow(frame_scores[top_indices].numpy(),
                   aspect="auto", origin="lower")
        plt.xlabel("Frame")
        plt.yticks(range(top_k), [DISPLAY_LABELS[i]
                   for i in top_indices.tolist()])
    
    highlight_top_labels(plt.gca(), [0, 1, 2])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_attention_chunk(attention, save_dir, prefix):
    att = attention[0, :, :, :].detach().cpu()
    head_sum = torch.sum(att, dim=0).numpy()
    plot_spec(head_sum, os.path.join(save_dir, f"{prefix}att_headsum.png"))
    for i in range(att.shape[0]):
        head = att[i].numpy()
        plot_spec(head, os.path.join(save_dir, f"{prefix}att_head{i}.png"))
        plt.imsave(
            fname=os.path.join(save_dir, f"{prefix}att_head{i}_imsave.png"),
            arr=head,
            format="png",
        )


def waveform_to_mel(wav, sample_rate):
    melspec_t = torchaudio.transforms.MelSpectrogram(
        sample_rate,
        f_min=60,
        f_max=min(7800, sample_rate // 2),
        hop_length=MEL_HOP_SAMPLES,
        win_length=1024,
        n_fft=1024,
        n_mels=64,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return to_db(melspec_t(wav))


def plot_telephony_conversion_comparison(original_wav, original_sr, converted_wav, converted_sr, save_path, original_codec_name):
    original_mel = waveform_to_mel(original_wav, original_sr)[
        0].detach().cpu().numpy()
    converted_mel = waveform_to_mel(converted_wav, converted_sr)[
        0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(original_mel, aspect="auto", origin="lower")
    axes[0].set_title(f"{original_codec_name} original decode")
    axes[0].set_ylabel("Mel Bin")

    axes[1].imshow(converted_mel, aspect="auto", origin="lower")
    axes[1].set_title("PCM waveform used by inference")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def get_audio_metadata(audio_path):
    try:
        return torchaudio.info(audio_path)
    except RuntimeError:
        return parse_riff_wave_metadata(audio_path)


def get_codec_name(metadata) -> str:
    encoding = getattr(metadata, "encoding", None)
    if encoding is None:
        return "unknown"
    return str(encoding)


def print_audio_metadata(audio_path, metadata):
    codec_name = get_codec_name(metadata)
    print(f"audio_path: {audio_path}")
    print(f"codec: {codec_name}")
    codec_tag = getattr(metadata, "codec_tag", None)
    if codec_tag is not None:
        print(f"codec_tag: {codec_tag}")
    print(f"sample_rate: {metadata.sample_rate}")
    print(f"num_frames: {metadata.num_frames}")
    print(f"num_channels: {metadata.num_channels}")
    print(f"bits_per_sample: {metadata.bits_per_sample}")


def maybe_save_converted_pcm(wav, save_path, target_sr):
    torchaudio.save(save_path, wav, sample_rate=target_sr,
                    encoding="PCM_S", bits_per_sample=16)
    print(f"saved decoded PCM wav to {save_path}")


def parse_riff_chunks(blob):
    offset = 12
    while offset + 8 <= len(blob):
        chunk_id = blob[offset:offset + 4]
        chunk_size = struct.unpack("<I", blob[offset + 4:offset + 8])[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        yield chunk_id, blob[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)


def parse_riff_wave_metadata(audio_path):
    blob = Path(audio_path).read_bytes()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise RuntimeError("unsupported audio file: not a RIFF/WAVE file")

    fmt_chunk = None
    data_chunk = None
    for chunk_id, chunk_body in parse_riff_chunks(blob):
        if chunk_id == b"fmt ":
            fmt_chunk = chunk_body
        elif chunk_id == b"data":
            data_chunk = chunk_body

    if fmt_chunk is None or data_chunk is None or len(fmt_chunk) < 16:
        raise RuntimeError("invalid RIFF/WAVE file: missing fmt or data chunk")

    format_tag, num_channels, sample_rate, avg_bytes_per_sec, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", fmt_chunk[:16]
    )
    encoding = {
        WAVE_FORMAT_PCM: "PCM_S",
        WAVE_FORMAT_ALAW: "PCM_ALAW",
        WAVE_FORMAT_MULAW: "PCM_MULAW",
        WAVE_FORMAT_VMS_G726: "G726_ADPCM",
    }.get(format_tag, f"UNKNOWN_0x{format_tag:04X}")

    if bits_per_sample > 0 and num_channels > 0:
        num_frames = len(data_chunk) // max(1,
                                            (bits_per_sample // 8) * num_channels)
    elif avg_bytes_per_sec > 0:
        num_frames = int(
            round(len(data_chunk) * sample_rate / avg_bytes_per_sec))
    else:
        num_frames = 0

    return AudioMetadata(
        sample_rate=sample_rate,
        num_frames=num_frames,
        num_channels=num_channels,
        bits_per_sample=bits_per_sample,
        encoding=encoding,
        codec_tag=f"0x{format_tag:04X}",
    )


def extract_wav_data_chunk(audio_path):
    blob = Path(audio_path).read_bytes()
    for chunk_id, chunk_body in parse_riff_chunks(blob):
        if chunk_id == b"data":
            return chunk_body
    raise RuntimeError("invalid RIFF/WAVE file: missing data chunk")


def decode_vms_g726_with_ffmpeg(audio_path, metadata):
    data_chunk = extract_wav_data_chunk(audio_path)
    decode_sample_rate = 8000
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "g726",
            "-ar",
            str(decode_sample_rate),
            "-ac",
            str(metadata.num_channels),
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        input=data_chunk,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    wav = torch.frombuffer(bytearray(process.stdout),
                           dtype=torch.float32).clone()
    wav = wav.reshape(-1, metadata.num_channels).transpose(0, 1).contiguous()
    return wav, decode_sample_rate


def load_audio_for_inference(audio_path, target_sr=16000, output_dir: Optional[str] = None):
    metadata = get_audio_metadata(audio_path)
    codec_name = get_codec_name(metadata)
    did_convert_telephony = False

    try:
        wav, sr = torchaudio.load(audio_path)
    except RuntimeError:
        codec_tag = getattr(metadata, "codec_tag", None)
        if codec_tag == "0x4148":
            wav, sr = decode_vms_g726_with_ffmpeg(audio_path, metadata)
            did_convert_telephony = True
            print(
                "decoded VMS telephony WAV (codec_tag 0x4148) with ffmpeg g726 fallback")
        else:
            raise RuntimeError(f"failed to decode audio with torchaudio; unsupported codec {codec_name}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    original_wav = wav.clone()
    original_sr = sr
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if output_dir is not None and did_convert_telephony:
        os.makedirs(output_dir, exist_ok=True)
        pcm_save_path = os.path.join(
            output_dir, "telephony_decoded_input_pcm.wav")
        maybe_save_converted_pcm(wav, pcm_save_path, target_sr)
        comparison_save_path = os.path.join(
            output_dir, "telephony_decode_comparison.png")
        plot_telephony_conversion_comparison(
            original_wav,
            original_sr,
            wav,
            target_sr,
            comparison_save_path,
            codec_name,
        )
        print(f"saved telephony decode comparison plot to { comparison_save_path}")
    return wav, sr, metadata


def plot_prediction_animation(prediction, save_dir, top_k=10, use_sec=False, seconds_per_frame=None, frame_step=5, figsize=(16, 6), save_dpi=300):
    frame_scores = prediction[0].detach().cpu()
    mean_scores = frame_scores.mean(dim=1)
    top_k = min(top_k, frame_scores.shape[0])
    top_indices = torch.topk(mean_scores, k=top_k).indices

    num_frames = frame_scores.shape[1]
    displayed_scores = frame_scores[top_indices].numpy()
    base_labels = [DISPLAY_LABELS[i] for i in top_indices.tolist()]

    os.makedirs(os.path.join(save_dir, "frames"), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    if use_sec:
        assert seconds_per_frame is not None
        duration_sec = num_frames * seconds_per_frame
        ax.imshow(
            displayed_scores,
            aspect="auto",
            origin="lower",
            extent=[0, duration_sec, 0, top_k],
        )
        line = ax.axvline(x=0, color="red", linestyle="-", linewidth=2)
        ax.set_xlabel("Seconds")
        ax.set_yticks(torch.arange(top_k).float() + 0.5)
        ax.set_yticklabels(base_labels)
    else:
        ax.imshow(displayed_scores, aspect="auto", origin="lower")
        line = ax.axvline(x=0, color="red", linestyle="-", linewidth=2)
        ax.set_xlabel("Frame")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(base_labels)

    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()

    frame_indices = list(range(0, num_frames, max(1, frame_step)))

    for output_frame_idx, i in enumerate(tqdm(frame_indices, desc="Saving animation frames")):
        if use_sec:
            line.set_xdata([i * seconds_per_frame, i * seconds_per_frame])
        else:
            line.set_xdata([i, i])

        frame_rank_positions = torch.topk(
            frame_scores[top_indices, i], k=min(3, top_k)
        ).indices.tolist()
        set_animated_y_labels(
            ax,
            base_labels,
            frame_rank_positions,
            frame_scores[top_indices, i].tolist(),
        )
        fig.savefig(
            os.path.join(save_dir, "frames", f"frame_{output_frame_idx:04d}.png"),
            dpi=save_dpi,
        )

    plt.close(fig)


def make_animated_video(wav_path, frames_dir, output_path, fps=25):
    base_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-i", wav_path,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-shortest",
    ]
    codec_variants = [
        ["-c:v", "libx264", "-c:a", "aac", "-b:a", "192k"],
        ["-c:v", "mpeg4", "-c:a", "aac", "-b:a", "192k"],
    ]

    last_error = None
    for codec_args in codec_variants:
        cmd = base_cmd + codec_args + [output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"saved animated video to {output_path}")
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            stderr = (exc.stderr or "").strip()
            if stderr:
                print(stderr)

    raise RuntimeError("ffmpeg failed to create the video") from last_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_attention", action="store_true")
    parser.add_argument("--plot_prediction", action="store_true")
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--use_sec", action="store_true")
    parser.add_argument("--frame_step", type=int, default=5)
    args = parser.parse_args()

    model = InferenceAudioSetStrong(args.ckpt_path)

    if args.audio_path is None:
        wav = torch.randn(1, 160000)
        metadata = None
    else:
        wav, original_sr, metadata = load_audio_for_inference(
            args.audio_path,
            target_sr=TARGET_SAMPLE_RATE,
            output_dir=args.output_dir,
        )
        print_audio_metadata(args.audio_path, metadata)
        if original_sr != 16000:
            print(f"resampled audio from {original_sr} Hz to 16000 Hz")

    with torch.no_grad():
        prediction = model.predict(wav)

    print("prediction shape:", prediction.shape)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.plot_prediction:
            if args.make_video:
                plot_prediction_animation(
                    prediction,
                    args.output_dir,
                    top_k=args.top_k,
                    use_sec=args.use_sec,
                    seconds_per_frame=model.seconds_per_prediction_frame,
                    frame_step=args.frame_step,
                )
                if args.audio_path is None:
                    print("Error: --make_video requires --audio_path")
                else:
                    make_animated_video(
                        args.audio_path,
                        os.path.join(args.output_dir, "frames"),
                        os.path.join(args.output_dir, "prediction_animation.mp4"),
                        fps=1 / (model.seconds_per_prediction_frame * max(1, args.frame_step))
                    )
            else:
                plot_path = os.path.join(args.output_dir, "prediction_topk.png")
                plot_prediction(
                    prediction,
                    plot_path,
                    top_k=args.top_k,
                    use_sec=args.use_sec,
                    seconds_per_frame=model.seconds_per_prediction_frame,
                )

        if args.plot_attention:
            with torch.no_grad():
                mel, mel_chunks, attentions = model.get_attention(wav)
            plot_spec(mel[0, 0].detach().cpu().numpy(),
                      os.path.join(args.output_dir, "mel.png"))
            for idx, mel_chunk in enumerate(mel_chunks):
                plot_spec(
                    mel_chunk[0, 0].detach().cpu().numpy(),
                    os.path.join(args.output_dir, f"mel_chunk{idx}.png"),
                )
                plot_attention_chunk(
                    attentions[idx], args.output_dir, prefix=f"chunk{idx}-")
