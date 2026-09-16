"""Create a reproducible, diverse, manually-labelled golden-set template.

The output intentionally contains no intent, escalation, or reply-quality labels.
Those columns are reserved for human annotation.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    thread_id: str
    message_text: str
    length_bucket: str
    position_bucket: str


def length_bucket(text: str) -> str:
    """Bucket by character count, including whitespace only at the edges."""
    size = len(text.strip())
    if size <= 80:
        return "short (0-80 chars)"
    if size <= 200:
        return "medium (81-200 chars)"
    return "long (201+ chars)"


def position_bucket(customer_index: int, customer_count: int) -> str:
    """Position among customer turns, not all turns, for a labellable message."""
    if customer_index == 0:
        return "first customer message"
    if customer_index == customer_count - 1:
        return "last customer message"
    return "middle customer message"


def read_candidates(input_path: Path) -> tuple[dict[tuple[str, str], list[Candidate]], int, int]:
    by_thread: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_read = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            by_thread[row["thread_id"]].append(row)

    strata: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for thread_id, turns in by_thread.items():
        customer_turns = sorted(
            (turn for turn in turns if turn["role"] == "customer"),
            key=lambda turn: int(turn["turn_order"]),
        )
        for index, turn in enumerate(customer_turns):
            candidate = Candidate(
                thread_id=thread_id,
                message_text=turn["text"],
                length_bucket=length_bucket(turn["text"]),
                position_bucket=position_bucket(index, len(customer_turns)),
            )
            strata[(candidate.length_bucket, candidate.position_bucket)].append(candidate)
    return strata, len(by_thread), rows_read


def choose_diverse_threads(
    strata: dict[tuple[str, str], list[Candidate]], count: int, seed: int
) -> list[Candidate]:
    """Take near-equal quotas from non-empty strata without reusing a thread."""
    rng = random.Random(seed)
    available = {key: list(values) for key, values in strata.items() if values}
    for values in available.values():
        rng.shuffle(values)
    keys = list(available)
    rng.shuffle(keys)
    unique_thread_count = len({candidate.thread_id for values in available.values() for candidate in values})
    target = min(count, unique_thread_count)
    if not keys or not target:
        return []

    base, remainder = divmod(target, len(keys))
    quotas = {key: base + (index < remainder) for index, key in enumerate(keys)}
    selected: list[Candidate] = []
    selected_threads: set[str] = set()

    # First pass aims for equal stratum quotas. A thread can appear in several
    # strata, so candidates from already selected threads are skipped.
    for key in keys:
        for candidate in available[key]:
            if len([item for item in selected if (item.length_bucket, item.position_bucket) == key]) >= quotas[key]:
                break
            if candidate.thread_id not in selected_threads:
                selected.append(candidate)
                selected_threads.add(candidate.thread_id)

    # Fill any quota shortfall from a random pool of remaining threads. This
    # preserves one labelling target per thread while still preferring diversity.
    remaining_by_thread: dict[str, list[Candidate]] = defaultdict(list)
    for values in available.values():
        for candidate in values:
            if candidate.thread_id not in selected_threads:
                remaining_by_thread[candidate.thread_id].append(candidate)
    remaining_threads = list(remaining_by_thread)
    rng.shuffle(remaining_threads)
    for thread_id in remaining_threads:
        if len(selected) >= target:
            break
        selected.append(rng.choice(remaining_by_thread[thread_id]))
        selected_threads.add(thread_id)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an unlabelled, stratified golden-set template.")
    parser.add_argument("--input", default="data/processed/uber_support_sample.csv")
    parser.add_argument("--output", default="evaluation/golden_set.csv")
    parser.add_argument("--threads", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Run subsample.py first.")

    print(f"Reading sample threads from {input_path} ...")
    strata, total_threads, rows_read = read_candidates(input_path)
    candidate_count = sum(len(values) for values in strata.values())
    print(f"Read {rows_read:,} turns across {total_threads:,} threads; found {candidate_count:,} customer messages.")
    selected = choose_diverse_threads(strata, args.threads, args.seed)
    if len(selected) < args.threads:
        print(f"Requested {args.threads:,} threads but only {len(selected):,} threads have a customer message.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["thread_id", "message_text", "true_intent", "should_escalate", "escalate_reason", "good_reply_notes"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in selected:
            # All annotation fields must remain blank for the human labeller.
            writer.writerow({
                "thread_id": candidate.thread_id,
                "message_text": candidate.message_text,
                "true_intent": "",
                "should_escalate": "",
                "escalate_reason": "",
                "good_reply_notes": "",
            })

    selected_summary = Counter((item.length_bucket, item.position_bucket) for item in selected)
    print(f"Wrote {len(selected):,} unlabelled threads to {output_path} (seed={args.seed}).")
    print("Selected-message strata:")
    for (message_length, position), size in sorted(selected_summary.items()):
        print(f"  {message_length}; {position}: {size:,}")


if __name__ == "__main__":
    main()
