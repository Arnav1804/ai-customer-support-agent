"""Filter a support account into complete, connected Twitter threads."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def linked_ids(row: dict[str, str]) -> set[str]:
    """Return tweet IDs referenced in either Kaggle relationship column."""
    ids: set[str] = set()
    for key in ("response_tweet_id", "in_response_to_tweet_id"):
        value = (row.get(key) or "").strip()
        if value:
            ids.update(part.strip() for part in value.split(",") if part.strip())
    return ids


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract every complete connected conversation involving one support account."
    )
    parser.add_argument("--input", default="data/raw/twcs.csv", help="Path to Kaggle twcs.csv")
    parser.add_argument("--author-id", required=True, help="Exact support account author_id, e.g. Uber_Support")
    parser.add_argument("--output", default="data/processed/uber_support.csv")
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Pass --input with the path to twcs.csv.")

    # First collect all brand tweets. The account ID check is exact but case-insensitive,
    # avoiding accidental inclusion of similarly named accounts.
    target_ids: set[str] = set()
    source_rows = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_rows += 1
            if row["author_id"].casefold() == args.author_id.casefold():
                target_ids.add(row["tweet_id"])
    if not target_ids:
        raise SystemExit(f"No tweets found for author_id '{args.author_id}'. Run inspect_brands.py first.")
    print(f"Scanned {source_rows:,} source rows and found {len(target_ids):,} {args.author_id} tweets.")
    print("Expanding complete connected threads ...")

    # The Kaggle reply fields form a graph. Repeated streaming passes compute the
    # connected-component closure without loading the entire source dataset into RAM.
    passes = 0
    while True:
        passes += 1
        before = len(target_ids)
        rows_scanned = 0
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_scanned += 1
                tweet_id = row["tweet_id"]
                neighbours = linked_ids(row)
                if tweet_id in target_ids or target_ids.intersection(neighbours):
                    target_ids.add(tweet_id)
                    target_ids.update(neighbours)
        added = len(target_ids) - before
        print(f"  pass {passes}: scanned {rows_scanned:,} rows; added {added:,} linked IDs ({len(target_ids):,} total)")
        if not added:
            break

    # Read only the selected component rows, then make connected-component IDs.
    selected: list[dict[str, str]] = []
    uf = UnionFind()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tweet_id = row["tweet_id"]
            if tweet_id in target_ids:
                selected.append(row)
                uf.find(tweet_id)
                for neighbour in linked_ids(row):
                    if neighbour in target_ids:
                        uf.union(tweet_id, neighbour)

    components: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        components[uf.find(row["tweet_id"])].append(row)

    # Use the smallest tweet ID in each component as a stable, human-auditable thread ID.
    rows_to_write: list[dict[str, str | int]] = []
    for thread in components.values():
        thread.sort(key=lambda row: (parse_timestamp(row["created_at"]), int(row["tweet_id"])))
        thread_id = "thread_" + min((row["tweet_id"] for row in thread), key=int)
        for turn_order, row in enumerate(thread, start=1):
            rows_to_write.append({
                "thread_id": thread_id,
                "turn_order": turn_order,
                "role": "brand" if row["author_id"].casefold() == args.author_id.casefold() else "customer",
                "text": row["text"],
                "created_at": row["created_at"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["thread_id", "turn_order", "role", "text", "created_at"])
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"Selected {len(selected):,} source rows and wrote {len(rows_to_write):,} turns across {len(components):,} complete threads to {output_path}.")


if __name__ == "__main__":
    main()
