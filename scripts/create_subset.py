#!/usr/bin/env python3
"""Create an AudioSet-style subset from AudioSet strong-label sources.

The output directory is the ROOT_PATH consumed by
scripts/dataset_preprocess/audioset_strong.bash:

|ROOT_PATH
|--data
|----train
|------*.wav
|----eval
|------*.wav
|--meta
|----source
|------audioset_train_strong.tsv
|------audioset_eval_strong.tsv
"""

import argparse
import csv
import shutil
import time
from pathlib import Path


DEFAULT_LABELS = "mid_street_surveillance_display_name.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description="Filter AudioSet strong data by MID list.")
    parser.add_argument(
        "root_path",
        type=Path,
        help="Output ROOT_PATH. This is the folder passed to audioset_strong.bash.",
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "huggingface"],
        default="kaggle",
        help="Input source type. Defaults to kaggle.",
    )
    parser.add_argument(
        "--audioset-path",
        type=Path,
        default=Path("kaggle"),
        help=(
            "Input folder containing audioset_train_strong.tsv, "
            "audioset_eval_strong.tsv, train_wav, and valid_wav."
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        default="enyoukai/AudioSet-Strong",
        help="Hugging Face dataset repo for --source huggingface.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("data/huggingface"),
        help="Hugging Face cache directory for --source huggingface.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token. Prefer setting HF_TOKEN/HUGGINGFACE_HUB_TOKEN "
            "instead of passing it on the command line."
        ),
    )
    parser.add_argument(
        "--hf-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset().",
    )
    parser.add_argument(
        "--hf-train-split",
        default="train",
        help="Hugging Face split name to export as train.",
    )
    parser.add_argument(
        "--hf-eval-split",
        default="test",
        help="Hugging Face split name to export as eval.",
    )
    parser.add_argument(
        "--hf-load-retries",
        type=int,
        default=30,
        help=(
            "Number of attempts for Hugging Face load_dataset(). Cached shards are "
            "reused between attempts. Defaults to 8."
        ),
    )
    parser.add_argument(
        "--hf-retry-wait",
        type=float,
        default=120.0,
        help="Seconds to wait between Hugging Face load_dataset() attempts.",
    )
    parser.add_argument(
        "--label-tsv",
        type=Path,
        default=Path(DEFAULT_LABELS),
        help="Two-column TSV of MID and display name to keep.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output folder.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write filtered source TSVs without copying wav files.",
    )
    parser.add_argument(
        "--no-validate-audio",
        action="store_true",
        help="Do not try to decode wav files before adding them to the output TSVs.",
    )
    return parser.parse_args()


def read_label_map(label_tsv):
    label_map = {}
    with label_tsv.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            mid = row[0].strip()
            display_name = row[1].strip()
            if mid:
                label_map[mid] = display_name
    if not label_map:
        raise ValueError(f"No labels found in {label_tsv}")
    return label_map


def can_decode_audio(path):
    try:
        import torchaudio

        torchaudio.load(str(path))
        return True
    except ImportError:
        try:
            import soundfile as sf

            sf.info(str(path))
            return True
        except Exception:
            return False
    except Exception:
        return False


def copy_wav(src, dst, overwrite):
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    shutil.copy2(src, dst)
    return True


def wav_candidates(wav_dir, segment_id):
    """Return likely source wav paths for a strong-TSV segment_id."""
    candidates = [wav_dir / f"{segment_id}.wav"]
    if "_" in segment_id:
        ytid = segment_id.rsplit("_", 1)[0]
        candidates.append(wav_dir / f"{ytid}.wav")
    return candidates


def copy_segment_wav(wav_dir, segment_id, data_dir, overwrite, validate_audio):
    dst_wav = data_dir / f"{segment_id}.wav"
    for src_wav in wav_candidates(wav_dir, segment_id):
        if not src_wav.exists():
            continue
        if validate_audio and not can_decode_audio(src_wav):
            return "invalid"
        if copy_wav(src_wav, dst_wav, overwrite):
            if validate_audio and not can_decode_audio(dst_wav):
                return "invalid"
            return "copied"
    return "missing"


def process_split(
    split_name,
    tsv_path,
    wav_dir,
    data_dir,
    source_tsv,
    label_map,
    overwrite,
    metadata_only,
    validate_audio,
):
    total_rows = 0
    matched_rows = 0
    copied_files = 0
    missing_files = 0
    invalid_files = 0
    event_rows = 0
    seen_copied = set()
    seen_missing = set()
    seen_invalid = set()

    with tsv_path.open(newline="") as input_file, source_tsv.open(
        "w", newline=""
    ) as tsv_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        writer = csv.DictWriter(
            tsv_file,
            fieldnames=["segment_id", "onset", "offset", "event_label"],
            delimiter="\t",
        )
        writer.writeheader()

        for row in reader:
            total_rows += 1
            segment_id = row["segment_id"].strip()
            label = row["label"].strip()
            if label not in label_map:
                continue

            matched_rows += 1
            if metadata_only:
                seen_copied.add(segment_id)
            elif segment_id not in seen_copied:
                copy_status = copy_segment_wav(
                    wav_dir, segment_id, data_dir, overwrite, validate_audio
                )
                if copy_status == "copied":
                    seen_copied.add(segment_id)
                    copied_files += 1
                elif copy_status == "invalid":
                    if segment_id not in seen_invalid:
                        seen_invalid.add(segment_id)
                        invalid_files += 1
                    continue
                else:
                    if segment_id not in seen_missing:
                        seen_missing.add(segment_id)
                        missing_files += 1
                    continue
            elif segment_id in seen_missing or segment_id in seen_invalid:
                continue

            writer.writerow(
                {
                    "segment_id": segment_id,
                    "onset": row["start_time_seconds"],
                    "offset": row["end_time_seconds"],
                    "event_label": label_map[label],
                }
            )
            event_rows += 1

    return {
        "split": split_name,
        "tsv_rows": total_rows,
        "matched_rows": matched_rows,
        "copied_files": copied_files,
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "event_rows": event_rows,
    }


def import_huggingface_dependencies():
    try:
        import numpy as np
        import soundfile as sf
        from datasets import Audio, load_dataset
        from huggingface_hub import login
    except ImportError as exc:
        raise ImportError(
            "Hugging Face mode requires datasets, huggingface_hub, numpy, and "
            "soundfile. Install them before using --source huggingface."
        ) from exc
    return Audio, load_dataset, login, np, sf


def first_present(row, names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_hf_labels(raw_labels):
    if raw_labels is None:
        return []
    if isinstance(raw_labels, str):
        return [raw_labels]
    if isinstance(raw_labels, dict):
        for key in ("mid", "mids", "label", "labels", "event_label", "event_labels"):
            if key in raw_labels:
                return normalize_hf_labels(raw_labels[key])
        return []
    labels = []
    try:
        iterator = iter(raw_labels)
    except TypeError:
        return [str(raw_labels)]
    for item in iterator:
        labels.extend(normalize_hf_labels(item))
    return labels


def label_display_name(label, label_map):
    if label in label_map:
        return label_map[label]
    if label in label_map.values():
        return label
    return None


def normalize_hf_events(row, label_map):
    events = first_present(row, ["events", "event", "annotations", "segments"])
    if not events:
        return []

    normalized = []
    for event in events:
        if not isinstance(event, dict):
            continue
        label = first_present(
            event,
            ["mid", "label", "event_label", "event_name", "human_label", "name"],
        )
        display_name = label_display_name(str(label), label_map) if label is not None else None
        if not display_name:
            continue
        onset = first_present(event, ["start", "onset", "start_time", "start_time_seconds"])
        offset = first_present(event, ["end", "offset", "end_time", "end_time_seconds"])
        if onset is None:
            onset = 0.0
        if offset is None:
            offset = first_present(event, ["duration", "audio_length"])
            if offset is None:
                offset = 10.0
        normalized.append((onset, offset, display_name))
    return normalized


def hf_audio_to_wav(audio, dst_wav, np, sf):
    if audio is None:
        return False
    if isinstance(audio, str):
        src = Path(audio)
        if src.exists():
            shutil.copy2(src, dst_wav)
            return True
        return False
    if isinstance(audio, dict):
        path = audio.get("path")
        if path and audio.get("array") is None and Path(path).exists():
            shutil.copy2(path, dst_wav)
            return True
        if audio.get("bytes") is not None:
            dst_wav.write_bytes(audio["bytes"])
            return True
        if audio.get("array") is not None:
            array = np.asarray(audio["array"])
            sampling_rate = int(audio.get("sampling_rate") or 16000)
            sf.write(dst_wav, array, sampling_rate)
            return True
    return False


def process_huggingface_split(
    split_name,
    dataset,
    data_dir,
    source_tsv,
    label_map,
    overwrite,
    metadata_only,
    validate_audio,
    np,
    sf,
):
    total_rows = 0
    matched_rows = 0
    copied_files = 0
    missing_files = 0
    invalid_files = 0
    event_rows = 0
    seen_copied = set()

    with source_tsv.open("w", newline="") as tsv_file:
        writer = csv.DictWriter(
            tsv_file,
            fieldnames=["segment_id", "onset", "offset", "event_label"],
            delimiter="\t",
        )
        writer.writeheader()

        for index, row in enumerate(dataset):
            total_rows += 1
            events = normalize_hf_events(row, label_map)
            if not events:
                labels = normalize_hf_labels(
                    first_present(
                        row,
                        ["mid", "mids", "label", "labels", "event_label", "event_labels"],
                    )
                )
                matched = [(label, label_display_name(label, label_map)) for label in labels]
                matched = [(label, display) for label, display in matched if display]
                events = []
                for _, display_name in matched:
                    events.append((None, None, display_name))
            if not events:
                continue

            segment_id = first_present(
                row, ["segment_id", "segment", "filename", "file_name", "ytid", "video_id"]
            )
            if segment_id is None:
                segment_id = f"{split_name}_{index:08d}"
            segment_id = Path(str(segment_id)).stem

            matched_rows += len(events)
            dst_wav = data_dir / f"{segment_id}.wav"
            if not metadata_only and segment_id not in seen_copied:
                if not dst_wav.exists() or overwrite:
                    audio = first_present(row, ["audio", "wav", "waveform"])
                    if not hf_audio_to_wav(audio, dst_wav, np, sf):
                        missing_files += 1
                        continue
                if validate_audio and not can_decode_audio(dst_wav):
                    invalid_files += 1
                    continue
                copied_files += 1
                seen_copied.add(segment_id)

            for onset, offset, display_name in events:
                if onset is None:
                    onset = first_present(row, ["start_time_seconds", "onset", "start", "start_time"])
                    if onset is None:
                        onset = 0.0
                if offset is None:
                    offset = first_present(row, ["end_time_seconds", "offset", "end", "end_time"])
                    if offset is None:
                        duration = first_present(row, ["duration", "audio_length"])
                        offset = duration if duration is not None else 10.0
                writer.writerow(
                    {
                        "segment_id": segment_id,
                        "onset": onset,
                        "offset": offset,
                        "event_label": display_name,
                    }
                )
                event_rows += 1

    return {
        "split": split_name,
        "tsv_rows": total_rows,
        "matched_rows": matched_rows,
        "copied_files": copied_files,
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "event_rows": event_rows,
    }


def load_huggingface_dataset(args, split_name):
    Audio, load_dataset, login, np, sf = import_huggingface_dependencies()
    if args.hf_token is not None:
        login(token=args.hf_token)
    attempts = max(1, args.hf_load_retries)
    for attempt in range(1, attempts + 1):
        try:
            dataset = load_dataset(
                args.hf_dataset,
                split=split_name,
                cache_dir=str(args.hf_cache_dir),
                trust_remote_code=args.hf_trust_remote_code,
            )
            break
        except Exception as exc:
            if attempt == attempts:
                raise
            print(
                (
                    f"load_dataset({args.hf_dataset!r}, split={split_name!r}) "
                    f"failed on attempt {attempt}/{attempts}: "
                    f"{type(exc).__name__}: {exc}; retrying in {args.hf_retry_wait:g}s"
                ),
                flush=True,
            )
            time.sleep(args.hf_retry_wait)
    for column in ("audio", "wav", "waveform"):
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Audio(decode=False))
            break
    return dataset, np, sf


def create_huggingface_subset(args, root_path, label_map):
    data_train = root_path / "data" / "train"
    data_eval = root_path / "data" / "eval"
    source_dir = root_path / "meta" / "source"
    data_train.mkdir(parents=True, exist_ok=True)
    data_eval.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for output_split, hf_split, data_dir, source_tsv in [
        ("train", args.hf_train_split, data_train, source_dir / "audioset_train_strong.tsv"),
        ("eval", args.hf_eval_split, data_eval, source_dir / "audioset_eval_strong.tsv"),
    ]:
        dataset, np, sf = load_huggingface_dataset(args, hf_split)
        summaries.append(
            process_huggingface_split(
                output_split,
                dataset,
                data_dir,
                source_tsv,
                label_map,
                args.overwrite,
                args.metadata_only,
                not args.no_validate_audio,
                np,
                sf,
            )
        )
    return summaries


def create_kaggle_subset(args, root_path, label_map):
    audioset_path = args.audioset_path.resolve()

    data_train = root_path / "data" / "train"
    data_eval = root_path / "data" / "eval"
    source_dir = root_path / "meta" / "source"
    data_train.mkdir(parents=True, exist_ok=True)
    data_eval.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        (
            "train",
            audioset_path / "audioset_train_strong.tsv",
            audioset_path / "train_wav",
            data_train,
            source_dir / "audioset_train_strong.tsv",
        ),
        (
            "eval",
            audioset_path / "audioset_eval_strong.tsv",
            audioset_path / "valid_wav",
            data_eval,
            source_dir / "audioset_eval_strong.tsv",
        ),
    ]

    summaries = []
    for split in splits:
        split_name, tsv_path, wav_dir, data_dir, source_tsv = split
        if not tsv_path.exists():
            raise FileNotFoundError(tsv_path)
        if not wav_dir.exists():
            raise FileNotFoundError(wav_dir)
        summaries.append(
            process_split(
                split_name,
                tsv_path,
                wav_dir,
                data_dir,
                source_tsv,
                label_map,
                args.overwrite,
                args.metadata_only,
                not args.no_validate_audio,
            )
        )

    return summaries


def main():
    args = parse_args()
    root_path = args.root_path.resolve()
    label_tsv = args.label_tsv.resolve()

    label_map = read_label_map(label_tsv)
    if args.source == "huggingface":
        summaries = create_huggingface_subset(args, root_path, label_map)
    else:
        summaries = create_kaggle_subset(args, root_path, label_map)

    print(f"Output ROOT_PATH: {root_path}")
    print(f"Source: {args.source}")
    print(f"Selected classes: {len(label_map)} from {label_tsv}")
    for summary in summaries:
        print(
            "{split}: tsv_rows={tsv_rows} matched_rows={matched_rows} "
            "copied_files={copied_files} missing_files={missing_files} "
            "invalid_files={invalid_files} event_rows={event_rows}".format(**summary)
        )


if __name__ == "__main__":
    main()
