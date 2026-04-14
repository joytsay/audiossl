
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
from torchvision import transforms

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
SECONDS_PER_FRAME = 160 / 16000





class InferenceAudioSetStrong(nn.Module):
    def __init__(self,ckpt_path):
        super().__init__()
        self.encoder = FrameAST_base()
        self.head = LinearHead(768, 407, use_norm=False, affine=False)
        self._load_ckpt(ckpt_path)
        self.transform = self._transform()

    def _transform(self):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        return transforms.Compose([melspec_t,
                                to_db,
                                normalize])

    
    def _load_ckpt(self,ckpt_path):
        s = torch.load(ckpt_path,map_location="cpu")
        state_dict = s["state_dict"]
        replaced_state_dict = {}
        for key in state_dict.keys():
            replaced_state_dict[key.replace("encoder.encoder","encoder")] =  state_dict[key]

        self.load_state_dict(replaced_state_dict)

    def _prepare_wav(self, wav):
        if len(wav.shape)==2:
            wav = wav.unsqueeze(1)
        else:
            assert len(wav.shape) == 3
        return wav

    def _chunk_mel(self, mel):
        chunk_len=1001 #10 secnods, consistent with the length of positional embedding
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        for i in range(num_chunks):
            start = i*chunk_len
            end = min((i+1) * chunk_len, total_len)
            if end > start:
                yield i, mel[:,:,:,start:end]

    def predict(self,wav):
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
            len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(wav.device)
            output_chunk = self.encoder.get_intermediate_layers(mel_chunk,len_chunk,n=1,scene=False)
            output.append(output_chunk)
        output=torch.cat(output,dim=1)
        output = self.head(output)
        return output

    def get_attention(self, wav):
        wav = self._prepare_wav(wav)
        mel = self.transform(wav)
        attentions = []
        mel_chunks = []
        for _, mel_chunk in self._chunk_mel(mel):
            attentions.append(self.encoder.get_last_selfattention(mel_chunk)[-1])
            mel_chunks.append(mel_chunk)
        return mel, mel_chunks, attentions


def plot_spec(x, save_path):
    plt.figure()
    plt.axis("off")
    plt.pcolormesh(x)
    plt.xticks([])
    plt.yticks([])
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=500, pad_inches=-0.01, transparent=True)
    plt.close()


def plot_prediction(prediction, save_path, top_k=10, use_sec=False):
    frame_scores = prediction[0].detach().cpu()
    mean_scores = frame_scores.mean(dim=1)
    top_k = min(top_k, frame_scores.shape[0])
    top_indices = torch.topk(mean_scores, k=top_k).indices

    plt.figure(figsize=(12, 4))
    if use_sec:
        num_frames = frame_scores.shape[1]
        duration_sec = num_frames * SECONDS_PER_FRAME
        plt.imshow(
            frame_scores[top_indices].numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, duration_sec, 0, top_k],
        )
        plt.xlabel("Seconds")
        plt.yticks(torch.arange(top_k).float() + 0.5, [DISPLAY_LABELS[i] for i in top_indices.tolist()])
    else:
        plt.imshow(frame_scores[top_indices].numpy(), aspect="auto", origin="lower")
        plt.xlabel("Frame")
        plt.yticks(range(top_k), [DISPLAY_LABELS[i] for i in top_indices.tolist()])
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


def load_audio_for_inference(audio_path, target_sr=16000):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_attention", action="store_true")
    parser.add_argument("--plot_prediction", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_sec", action="store_true")
    args = parser.parse_args()

    model = InferenceAudioSetStrong(args.ckpt_path)

    if args.audio_path is None:
        wav = torch.randn(1,160000)
    else:
        wav, original_sr = load_audio_for_inference(args.audio_path)
        if original_sr != 16000:
            print(f"resampled audio from {original_sr} Hz to 16000 Hz")

    with torch.no_grad():
        prediction = model.predict(wav)

    print("prediction shape:", prediction.shape)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.plot_prediction:
            plot_prediction(
                prediction,
                os.path.join(args.output_dir, "prediction_topk.png"),
                top_k=args.top_k,
                use_sec=args.use_sec,
            )

        if args.plot_attention:
            with torch.no_grad():
                mel, mel_chunks, attentions = model.get_attention(wav)
            plot_spec(mel[0, 0].detach().cpu().numpy(), os.path.join(args.output_dir, "mel.png"))
            for idx, mel_chunk in enumerate(mel_chunks):
                plot_spec(
                    mel_chunk[0, 0].detach().cpu().numpy(),
                    os.path.join(args.output_dir, f"mel_chunk{idx}.png"),
                )
                plot_attention_chunk(attentions[idx], args.output_dir, prefix=f"chunk{idx}-")
