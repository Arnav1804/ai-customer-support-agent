"""List outbound support-account author IDs matching a brand name."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find outbound support-account IDs matching a brand name."
    )
    parser.add_argument("--input", default="data/raw/twcs.csv", help="Path to Kaggle twcs.csv")
    parser.add_argument("--brand", default="uber", help="Case-insensitive text to match")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Pass --input with the path to twcs.csv.")

    print(f"Inspecting outbound authors in {input_path} ...")
    counts: Counter[str] = Counter()
    rows_read = outbound_rows = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            if row["inbound"].strip().casefold() == "false":
                outbound_rows += 1
                counts[row["author_id"]] += 1

    query = args.brand.casefold()
    matches = [(author, volume) for author, volume in counts.items() if query in author.casefold()]
    matches.sort(key=lambda item: (-item[1], item[0].casefold()))
    print(f"Read {rows_read:,} rows; found {outbound_rows:,} outbound tweets from {len(counts):,} authors.")
    if matches:
        print(f"Authors matching '{args.brand}':")
        for author, volume in matches:
            print(f"  {author}: {volume:,} tweets")
    else:
        print(f"No outbound author_id values matched '{args.brand}'.")


if __name__ == "__main__":
    main()
