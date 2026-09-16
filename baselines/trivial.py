"""Label-dependent trivial baselines for the hand-annotated golden set."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


OUTPUT_COLUMNS = ["thread_id", "predicted_intent", "predicted_escalate", "predicted_reason"]


def most_common_intent(rows: list[dict[str, str]]) -> tuple[str, Counter[str]]:
    labels = Counter(row["true_intent"].strip() for row in rows if row.get("true_intent", "").strip())
    if not labels:
        raise SystemExit(
            "No true_intent labels found. Fill in the hand-labelled golden_set.csv before running trivial.py."
        )
    # Explicit alphabetical tie-break keeps predictions reproducible.
    intent = sorted(labels, key=lambda label: (-labels[label], label.casefold()))[0]
    return intent, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the annotated majority intent and always escalate.")
    parser.add_argument("--input", default="evaluation/golden_set.csv")
    parser.add_argument("--output", default="outputs/trivial_predictions.csv")
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Build and hand-label the golden set first.")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {input_path}.")
    majority_intent, labels = most_common_intent(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "thread_id": row["thread_id"],
                "predicted_intent": majority_intent,
                "predicted_escalate": "escalate",
                "predicted_reason": "Trivial baseline: always escalate every message.",
            })

    print(f"Read {len(rows):,} golden-set rows; {sum(labels.values()):,} have a true_intent label.")
    print(f"Majority intent: {majority_intent} ({labels[majority_intent]:,} labelled rows)")
    print(f"Predicted intent distribution: {majority_intent}: {len(rows):,}")
    print(f"Predicted escalation distribution: escalate: {len(rows):,}")
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
