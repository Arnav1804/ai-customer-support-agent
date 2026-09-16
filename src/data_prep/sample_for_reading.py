"""Print a reproducible reading sample of first customer messages by thread."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print random first customer messages, one per conversation thread."
    )
    parser.add_argument("--input", default="data/processed/uber_support_sample.csv")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for a repeatable reading order")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Run subsample.py first.")

    # The filtered file is written in turn order. Still, compare turn_order so
    # this remains correct if a caller reorders the CSV before using this script.
    first_customer: dict[str, tuple[int, str]] = {}
    rows_read = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            if row["role"] != "customer":
                continue
            thread_id = row["thread_id"]
            turn_order = int(row["turn_order"])
            if thread_id not in first_customer or turn_order < first_customer[thread_id][0]:
                first_customer[thread_id] = (turn_order, row["text"])

    candidates = sorted(first_customer.items())
    sample_size = min(args.count, len(candidates))
    if not candidates:
        raise SystemExit("No customer turns found in the input file.")
    if sample_size < args.count:
        print(f"Requested {args.count} messages but only {sample_size} threads have a customer message.")

    chooser = random.Random(args.seed)
    selected = chooser.sample(candidates, sample_size)
    seed_note = f" with seed={args.seed}" if args.seed is not None else ""
    print(f"Read {rows_read:,} turns from {input_path}; found {len(candidates):,} first customer messages.")
    print(f"Printing {sample_size} random messages{seed_note}:\n")
    for index, (thread_id, (turn_order, text)) in enumerate(selected, start=1):
        print(f"[{index}] {thread_id} (turn {turn_order})")
        print(text)
        print()


if __name__ == "__main__":
    main()
