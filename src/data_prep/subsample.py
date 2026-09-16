"""Take a reproducible random sample of complete support threads."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded sample of complete conversation threads.")
    parser.add_argument("--input", default="data/processed/uber_support.csv")
    parser.add_argument("--output", default="data/processed/uber_support_sample.csv")
    parser.add_argument("--threads", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Run filter_brand.py first.")
    print(f"Reading complete threads from {input_path} ...")
    by_thread: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            by_thread[row["thread_id"]].append(row)

    thread_ids = sorted(by_thread)
    requested = args.threads
    sample_size = min(requested, len(thread_ids))
    if sample_size < requested:
        print(f"Requested {requested:,} threads but only {len(thread_ids):,} exist; using all available threads.")
    selected = set(random.Random(args.seed).sample(thread_ids, sample_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for thread_id in thread_ids:
            if thread_id in selected:
                writer.writerows(by_thread[thread_id])
                rows_written += len(by_thread[thread_id])
    print(f"Total threads in filtered file: {len(thread_ids):,}")
    print(f"Sampled threads (seed={args.seed}): {sample_size:,}")
    print(f"Wrote {rows_written:,} turns to {output_path}")


if __name__ == "__main__":
    main()
