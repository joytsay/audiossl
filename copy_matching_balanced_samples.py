#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy N random audio files from matching audioSetStrongBalanced class "
            "folders listed in a TSV, and optionally also copy from a matching "
            "SED class-folder tree if it exists."
        )
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("audioSetStrongSED/mid_street_surveillance_display_name.tsv"),
        help="TSV file with two columns: MID and display name.",
    )
    parser.add_argument(
        "--balanced-root",
        type=Path,
        default=Path("audioSetStrongBalanced"),
        help="Root directory containing Balanced class folders.",
    )
    parser.add_argument(
        "--sed-root",
        type=Path,
        default=Path("audioSetStrongSED/mid_street_surveillance_display_name"),
        help="Optional root directory containing SED class folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("sampled_mid_street_surveillance_display_name"),
        help="Output directory for copied samples.",
    )
    parser.add_argument(
        "-n",
        "--num-per-class",
        type=int,
        default=2,
        help="Number of random files to copy from each matching class folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--copy-sed",
        action="store_true",
        help="Also copy from the matching SED folder tree if it exists.",
    )
    return parser.parse_args()


def load_display_names(tsv_path: Path) -> list[str]:
    display_names: list[str] = []
    with tsv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            display_name = row[1].strip()
            if display_name:
                display_names.append(display_name)
    return display_names


def list_audio_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )


def copy_random_files(src_dir: Path, dst_dir: Path, num_per_class: int) -> int:
    files = list_audio_files(src_dir)
    if not files:
        return 0

    chosen = random.sample(files, k=min(num_per_class, len(files)))
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_path in chosen:
        shutil.copy2(src_path, dst_dir / src_path.name)

    return len(chosen)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def tokenize_name(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.casefold()) if token}


def build_folder_lookup(root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        lookup.setdefault(normalize_name(path.name), path)
    return lookup


def resolve_class_dir(class_name: str, folder_lookup: dict[str, Path]) -> Path | None:
    normalized = normalize_name(class_name)
    exact_match = folder_lookup.get(normalized)
    if exact_match is not None:
        return exact_match

    substring_matches = [
        path
        for folder_name, path in folder_lookup.items()
        if normalized in folder_name or folder_name in normalized
    ]
    if substring_matches:
        substring_matches.sort(key=lambda path: len(normalize_name(path.name)), reverse=True)
        if len(substring_matches) == 1 or len(normalize_name(substring_matches[0].name)) > len(
            normalize_name(substring_matches[1].name)
        ):
            return substring_matches[0]

    class_tokens = tokenize_name(class_name)
    best_score = 0
    best_match: Path | None = None
    tie = False

    for path in folder_lookup.values():
        score = len(class_tokens & tokenize_name(path.name))
        if score > best_score:
            best_score = score
            best_match = path
            tie = False
        elif score == best_score and score > 0:
            tie = True

    if best_score > 0 and not tie:
        return best_match

    return None


def copy_tree_samples(
    class_names: list[str],
    src_root: Path,
    dst_root: Path,
    num_per_class: int,
    label: str,
) -> None:
    folder_lookup = build_folder_lookup(src_root)

    for class_name in class_names:
        src_dir = resolve_class_dir(class_name, folder_lookup)
        dst_dir = dst_root / class_name

        if src_dir is None:
            print(f"[{label}] {class_name}: folder not found")
            continue

        copied = copy_random_files(src_dir, dst_dir, num_per_class)
        print(f"[{label}] {class_name}: copied {copied}")


def main() -> None:
    args = parse_args()

    if args.num_per_class < 1:
        raise SystemExit("--num-per-class must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    class_names = load_display_names(args.tsv)

    copy_tree_samples(
        class_names=class_names,
        src_root=args.balanced_root,
        dst_root=args.output_root,
        num_per_class=args.num_per_class,
        label="Balanced",
    )

    if args.copy_sed:
        copy_tree_samples(
            class_names=class_names,
            src_root=args.sed_root,
            dst_root=args.output_root / "audioSetStrongSED" / args.sed_root.name,
            num_per_class=args.num_per_class,
            label="SED",
        )


if __name__ == "__main__":
    main()
