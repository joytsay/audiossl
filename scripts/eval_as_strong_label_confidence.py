import argparse
import csv
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audiossl.transforms.common import MinMax


TARGET_SAMPLE_RATE = 16000
MEL_HOP_SAMPLES = 160


def load_frame_ast_base():
    if "pytorch_lightning" not in sys.modules:
        lightning = types.ModuleType("pytorch_lightning")
        lightning.LightningModule = nn.Module
        sys.modules["pytorch_lightning"] = lightning
    if "transformers.optimization" not in sys.modules:
        transformers = types.ModuleType("transformers")
        optimization = types.ModuleType("transformers.optimization")
        optimization.AdamW = torch.optim.AdamW
        transformers.optimization = optimization
        sys.modules["transformers"] = transformers
        sys.modules["transformers.optimization"] = optimization

    module_path = (
        REPO_ROOT
        / "audiossl/methods/atstframe/downstream/comparison_models/models/frame_atst.py"
    )
    spec = importlib.util.spec_from_file_location("frame_atst_local", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FrameAST_base


FrameAST_base = load_frame_ast_base()


class LinearHead(nn.Module):
    def __init__(self, dim, num_labels=407):
        super().__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, temp=1):
        strong = self.sigmoid(self.linear(x) / temp)
        return strong.transpose(1, 2)


class AudioSetStrongModel(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.encoder = FrameAST_base()
        self.head = LinearHead(768, 407)
        self._load_ckpt(ckpt_path)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            TARGET_SAMPLE_RATE,
            f_min=60,
            f_max=7800,
            hop_length=MEL_HOP_SAMPLES,
            win_length=1024,
            n_fft=1024,
            n_mels=64,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.normalize = MinMax(min=-79.6482, max=50.6842)

    def _load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        remapped = {
            key.replace("encoder.encoder", "encoder"): value
            for key, value in state_dict.items()
        }
        self.load_state_dict(remapped)

    def forward(self, wav):
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        mel = self.normalize(self.to_db(self.melspec(wav)))
        chunks = []
        chunk_len = 1001
        total_len = mel.shape[-1]
        for start in range(0, total_len, chunk_len):
            mel_chunk = mel[:, :, :, start : start + chunk_len]
            lengths = torch.tensor([mel_chunk.shape[-1]], device=mel.device).expand(
                mel.shape[0]
            )
            chunks.append(
                self.encoder.get_intermediate_layers(
                    mel_chunk, lengths, n=1, scene=False
                )
            )
        return self.head(torch.cat(chunks, dim=1))


def read_mid_list(path):
    mids = []
    mid_to_name = {}
    with Path(path).open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            mid = row[0].strip()
            name = row[1].strip() if len(row) > 1 else mid
            mids.append(mid)
            mid_to_name[mid] = name
    return mids, mid_to_name


def read_common_labels(path):
    with Path(path).open() as handle:
        return [line.strip() for line in handle if line.strip()]


def read_mid_to_display_name(path):
    mid_to_name = {}
    with Path(path).open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2:
                mid_to_name[row[0]] = row[1]
    return mid_to_name


def display_labels_for_common_labels(common_labels, mid_to_display_name_path):
    mid_to_name = read_mid_to_display_name(mid_to_display_name_path)
    return [mid_to_name.get(mid, mid) for mid in common_labels]


def read_class_label_names(path):
    label_to_mid = {}
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_to_mid[row["display_name"]] = row["mid"]
    return label_to_mid


def read_eval_truth(path, selected_mids, selected_names, class_labels_path=None):
    truth = {}
    selected_mids = set(selected_mids)
    selected_names = set(selected_names)
    label_to_mid = {}
    if class_labels_path is not None and Path(class_labels_path).exists():
        label_to_mid = read_class_label_names(class_labels_path)

    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            filename = row["filename"]
            label = row["event_label"]
            mid = label_to_mid.get(label)
            if mid in selected_mids:
                truth.setdefault(filename, set()).add(mid)
            elif label in selected_names:
                truth.setdefault(filename, set()).add(label)
    return truth


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
    return wav


def summarize_prediction(prediction, label_indices):
    scores = prediction[0].detach().cpu()
    rows = []
    for label_idx in label_indices:
        label_scores = scores[label_idx]
        rows.append(
            {
                "mean_confidence": float(label_scores.mean().item()),
                "max_confidence": float(label_scores.max().item()),
                "p95_confidence": float(torch.quantile(label_scores, 0.95).item()),
            }
        )
    return rows


def median_filter_time(binary_preds, kernel_size):
    if kernel_size <= 1:
        return binary_preds
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    padded = F.pad(binary_preds.float(), (pad_left, pad_right), mode="replicate")
    windows = padded.unfold(-1, kernel_size, 1)
    return windows.median(dim=-1).values


def decode_strong_predictions(prediction, thresholds, filenames, labels, median_kernel=7):
    frames_per_second = TARGET_SAMPLE_RATE / MEL_HOP_SAMPLES / 4
    decoded = {
        threshold: [] for threshold in thresholds
    }
    scores = prediction.detach()
    for threshold in thresholds:
        binary = median_filter_time(scores > threshold, median_kernel).bool()
        for batch_idx, filename in enumerate(filenames):
            for class_idx, label in enumerate(labels):
                active = binary[batch_idx, class_idx]
                padded = F.pad(active.to(torch.int8), (1, 1))
                changes = padded[1:] - padded[:-1]
                onsets = torch.nonzero(changes == 1, as_tuple=False).flatten()
                offsets = torch.nonzero(changes == -1, as_tuple=False).flatten()
                for onset, offset in zip(onsets.tolist(), offsets.tolist()):
                    decoded[threshold].append(
                        {
                            "filename": Path(filename).name,
                            "event_label": label,
                            "onset": onset / frames_per_second,
                            "offset": offset / frames_per_second,
                        }
                    )
    return {
        threshold: pd.DataFrame(
            rows, columns=["filename", "event_label", "onset", "offset"]
        )
        for threshold, rows in decoded.items()
    }


def merge_prediction_dfs(target, source):
    for threshold, dataframe in source.items():
        if dataframe.empty:
            continue
        target[threshold] = pd.concat([target[threshold], dataframe], ignore_index=True)


def write_filtered_eval_files(eval_tsv, durations_tsv, filenames, out_dir):
    filenames = {Path(filename).name for filename in filenames}
    filtered_eval_path = out_dir / "psds_eval.tsv"
    filtered_durations_path = out_dir / "psds_eval_durations.tsv"

    eval_df = pd.read_csv(eval_tsv, sep="\t")
    eval_df = eval_df[eval_df["filename"].isin(filenames)]
    eval_df.to_csv(filtered_eval_path, sep="\t", index=False)

    durations_df = pd.read_csv(durations_tsv, sep="\t")
    durations_df = durations_df[durations_df["filename"].isin(filenames)]
    durations_df.to_csv(filtered_durations_path, sep="\t", index=False)

    return filtered_eval_path, filtered_durations_path


def compute_psds_metrics(prediction_dfs, eval_tsv, durations_tsv, out_dir):
    from audiossl.methods.atstframe.downstream.utils_psds_eval import evaluation
    from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import (
        compute_per_intersection_macro_f1,
        compute_psds_from_operating_points,
    )

    add_op = evaluation.PSDSEval.add_operating_point_single_thread
    if "weighted" not in inspect.signature(add_op).parameters:
        def add_operating_point_weighted_compat(self, det, info=None, weighted=False):
            return add_op(self, det, info=info)

        evaluation.PSDSEval.add_operating_point_single_thread = (
            add_operating_point_weighted_compat
        )

    psds_fn = evaluation.PSDSEval.psds
    if "weighted" not in inspect.signature(psds_fn).parameters:
        def psds_weighted_compat(
            self,
            alpha_ct=0,
            alpha_st=0,
            max_efpr=100,
            weighted=False,
        ):
            return psds_fn(
                self,
                alpha_ct=alpha_ct,
                alpha_st=alpha_st,
                max_efpr=max_efpr,
            )

        evaluation.PSDSEval.psds = psds_weighted_compat

    psds_dir = out_dir / "psds"
    scenario1_dfs = {
        threshold: dataframe.copy() for threshold, dataframe in prediction_dfs.items()
    }
    scenario2_dfs = {
        threshold: dataframe.copy() for threshold, dataframe in prediction_dfs.items()
    }
    f1_dfs = {
        threshold: dataframe.copy() for threshold, dataframe in prediction_dfs.items()
    }
    scenario1 = compute_psds_from_operating_points(
        scenario1_dfs,
        str(eval_tsv),
        str(durations_tsv),
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0,
        alpha_st=0.0,
        save_dir=str(psds_dir / "scenario1"),
        weighted=False,
    )
    scenario2 = compute_psds_from_operating_points(
        scenario2_dfs,
        str(eval_tsv),
        str(durations_tsv),
        dtc_threshold=0.1,
        gtc_threshold=0.1,
        cttc_threshold=0.3,
        alpha_ct=0.5,
        alpha_st=0.0,
        save_dir=str(psds_dir / "scenario2"),
        weighted=False,
    )
    mid_threshold = list(prediction_dfs.keys())[len(prediction_dfs) // 2]
    intersection_f1 = compute_per_intersection_macro_f1(
        {"0.5": f1_dfs[mid_threshold]},
        str(eval_tsv),
        str(durations_tsv),
    )
    return {
        "psds1": float(scenario1),
        "psds2": float(scenario2),
        "intersection_f1_macro_percent": float(intersection_f1 * 100),
        "f1_threshold": float(mid_threshold),
    }


def make_totals(selected):
    return {
        name: {
            "n": 0,
            "mean_sum": 0.0,
            "max_sum": 0.0,
            "p95_sum": 0.0,
            "truth_n": 0,
            "truth_max_sum": 0.0,
        }
        for _, _, name, _ in selected
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/atst_ft_StrongAS_eps28.ckpt")
    parser.add_argument("--audio-dir", default="AudioSet_strong/data/eval")
    parser.add_argument("--label-tsv", default="mid_street_surveillance_display_name.tsv")
    parser.add_argument("--common-labels", default="common_labels.txt")
    parser.add_argument("--mid-to-display-name", default="mid_to_display_name.tsv")
    parser.add_argument("--eval-tsv", default="AudioSet_strong/meta/eval/eval.tsv")
    parser.add_argument("--eval-durations", default="AudioSet_strong/meta/eval/eval_durations.tsv")
    parser.add_argument("--class-labels", default="AudioSet_strong/class_labels_indices.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-psds", action="store_true")
    parser.add_argument("--psds-thresholds", type=int, default=50)
    parser.add_argument("--median-kernel", type=int, default=7)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write only the top-k selected labels per file, ranked by max confidence.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of selected labels to write per file when --compact is set.",
    )
    args = parser.parse_args()
    if args.topk < 1:
        raise SystemExit("--topk must be >= 1")

    selected_mids, mid_to_name = read_mid_list(args.label_tsv)
    common_labels = read_common_labels(args.common_labels)
    display_labels = display_labels_for_common_labels(
        common_labels, args.mid_to_display_name
    )
    mid_to_index = {mid: idx for idx, mid in enumerate(common_labels)}
    missing = [mid for mid in selected_mids if mid not in mid_to_index]
    if missing:
        raise SystemExit(f"MIDs are not in {args.common_labels}: {missing}")

    selected = [
        (seq, mid, mid_to_name[mid], mid_to_index[mid])
        for seq, mid in enumerate(selected_mids, start=1)
        if mid in mid_to_index
    ]
    truth = read_eval_truth(
        args.eval_tsv,
        [mid for _, mid, _, _ in selected],
        [name for _, _, name, _ in selected],
        args.class_labels,
    )

    audio_dir = Path(args.audio_dir)
    out_dir = audio_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file_label_confidence.csv"
    summary_path = out_dir / "summary_label_confidence.csv"

    model = AudioSetStrongModel(args.ckpt).to(args.device).eval()
    wav_paths = sorted(audio_dir.rglob("*.wav"))
    if args.limit is not None:
        wav_paths = wav_paths[: args.limit]
    thresholds = [
        1 / (args.psds_thresholds * 2) + i / args.psds_thresholds
        for i in range(args.psds_thresholds)
    ]
    psds_prediction_dfs = {
        threshold: pd.DataFrame(columns=["filename", "event_label", "onset", "offset"])
        for threshold in thresholds
    }

    per_file_fieldnames = [
        "filename",
        "mid",
        "display_name",
        "mean_confidence",
        "max_confidence",
        "p95_confidence",
        "is_ground_truth_label",
    ]
    summary_fieldnames = [
        "display_name",
        "n_files",
        "avg_mean_confidence",
        "avg_max_confidence",
        "avg_p95_confidence",
        "n_ground_truth_files",
        "avg_max_confidence_on_ground_truth",
    ]

    totals = make_totals(selected)

    with per_file_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_file_fieldnames)
        writer.writeheader()
        for file_idx, wav_path in enumerate(wav_paths, start=1):
            if file_idx == 1 or file_idx % 100 == 0 or file_idx == len(wav_paths):
                print(f"scoring {file_idx}/{len(wav_paths)}: {wav_path}")
            output_filename = str(wav_path.relative_to(audio_dir))
            wav = load_audio(wav_path).to(args.device)
            with torch.no_grad():
                prediction = model(wav)
            if args.compute_psds:
                decoded = decode_strong_predictions(
                    prediction,
                    thresholds,
                    [wav_path.name],
                    display_labels,
                    median_kernel=args.median_kernel,
                )
                merge_prediction_dfs(psds_prediction_dfs, decoded)
            rows = summarize_prediction(prediction, [idx for _, _, _, idx in selected])
            filename_truth = truth.get(wav_path.name, set())
            file_rows = []
            for (_, mid, name, _), row in zip(selected, rows):
                is_truth = mid in filename_truth or name in filename_truth
                output_row = {
                    "filename": output_filename,
                    "mid": mid,
                    "display_name": name,
                    **row,
                    "is_ground_truth_label": int(is_truth),
                }
                file_rows.append(output_row)
                totals[name]["n"] += 1
                totals[name]["mean_sum"] += row["mean_confidence"]
                totals[name]["max_sum"] += row["max_confidence"]
                totals[name]["p95_sum"] += row["p95_confidence"]
                if is_truth:
                    totals[name]["truth_n"] += 1
                    totals[name]["truth_max_sum"] += row["max_confidence"]
            if args.compact:
                file_rows = sorted(
                    file_rows,
                    key=lambda row: row["max_confidence"],
                    reverse=True,
                )[: args.topk]
            for row in file_rows:
                writer.writerow(row)

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for _, _, name, _ in selected:
            total = totals[name]
            n = max(total["n"], 1)
            truth_n = total["truth_n"]
            writer.writerow(
                {
                    "display_name": name,
                    "n_files": total["n"],
                    "avg_mean_confidence": total["mean_sum"] / n,
                    "avg_max_confidence": total["max_sum"] / n,
                    "avg_p95_confidence": total["p95_sum"] / n,
                    "n_ground_truth_files": truth_n,
                    "avg_max_confidence_on_ground_truth": (
                        total["truth_max_sum"] / truth_n if truth_n else ""
                    ),
                }
            )

    print(f"wrote {per_file_path}")
    print(f"wrote {summary_path}")

    if args.compute_psds:
        filtered_eval_tsv, filtered_durations_tsv = write_filtered_eval_files(
            args.eval_tsv, args.eval_durations, [path.name for path in wav_paths], out_dir
        )
        metrics = compute_psds_metrics(
            psds_prediction_dfs, filtered_eval_tsv, filtered_durations_tsv, out_dir
        )
        metrics_path = out_dir / "psds_metrics.json"
        with metrics_path.open("w") as handle:
            json.dump(metrics, handle, indent=2)
        metrics_csv_path = out_dir / "psds_metrics.csv"
        with metrics_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(json.dumps(metrics, indent=2))
        print(f"wrote {metrics_path}")
        print(f"wrote {metrics_csv_path}")


if __name__ == "__main__":
    main()
