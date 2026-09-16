"""CLI entry point for the classifier, retrieval, grounded reply, and escalation stages."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from classifier.intent_classifier import classify_message
from common.llm import LLMError, OpenAICompatibleClient, UnavailableClient
from escalation.decide import decide_escalation
from generation.reply_generator import draft_reply
from retrieval.similar_threads import ThreadRetriever


PREDICTION_COLUMNS = [
    "predicted_intent", "classifier_confidence", "predicted_escalate", "predicted_reason",
    "drafted_reply", "retrieved_thread_ids", "pipeline_status",
]


def run_message(message: str, retriever: ThreadRetriever, client, thread_id: str | None = None) -> dict[str, str | float]:
    """Run all four stages for one message; designed for direct unit testing with a fake client."""
    intent_prediction = classify_message(message, client)
    retrieved = retriever.retrieve(message, top_k=3, exclude_thread_id=thread_id)
    reply = draft_reply(message, intent_prediction.intent, retrieved, client)
    decision = decide_escalation(message, intent_prediction.intent, intent_prediction.confidence, reply)
    return {
        "predicted_intent": intent_prediction.intent,
        "classifier_confidence": round(intent_prediction.confidence, 4),
        "predicted_escalate": decision.action,
        "predicted_reason": decision.reason,
        "drafted_reply": reply,
        "retrieved_thread_ids": ";".join(item.thread_id for item in retrieved),
        "pipeline_status": "completed",
    }


def skipped_result(reason: str) -> dict[str, str]:
    return {
        "predicted_intent": "", "classifier_confidence": "", "predicted_escalate": "",
        "predicted_reason": "", "drafted_reply": "", "retrieved_thread_ids": "",
        "pipeline_status": f"skipped: {reason}",
    }


def message_from_row(row: dict[str, str]) -> str:
    message = row.get("message_text") or row.get("text")
    if not message:
        raise ValueError("Input CSV needs a non-empty message_text column (or text column).")
    return message


def run_batch(input_path: Path, output_path: Path, retriever: ThreadRetriever, client) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = input_columns + [column for column in PREDICTION_COLUMNS if column not in input_columns]
    completed = skipped = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row_number, row in enumerate(rows, start=2):
            try:
                result = run_message(message_from_row(row), retriever, client, row.get("thread_id") or None)
                completed += 1
            except (LLMError, ValueError) as exc:
                # The API client already retries once. Preserve the row and continue the batch.
                logging.warning("Skipping input row %d after stage failure: %s", row_number, exc)
                result = skipped_result(str(exc))
                skipped += 1
            writer.writerow({**row, **result})
    print(f"Processed {len(rows):,} rows: {completed:,} completed, {skipped:,} skipped.")
    print(f"Wrote pipeline predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Uber Support pipeline on one message or a CSV batch.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="One customer message to process")
    source.add_argument("--input", help="CSV containing message_text (or text) and optionally thread_id")
    parser.add_argument("--sample", default="data/processed/uber_support_sample.csv", help="Seeded historical-thread corpus")
    parser.add_argument("--output", default="outputs/pipeline_predictions.csv", help="Batch CSV output path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        retriever = ThreadRetriever.from_csv(args.sample)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not build retriever from {args.sample}: {exc}") from exc
    try:
        client = OpenAICompatibleClient.from_environment()
    except LLMError as exc:
        # Batch runs remain inspectable without a configured API, producing skipped rows.
        logging.warning("LLM is unavailable: %s", exc)
        client = UnavailableClient(str(exc))

    if args.message:
        try:
            print(json.dumps(run_message(args.message, retriever, client), ensure_ascii=False, indent=2))
        except (LLMError, ValueError) as exc:
            raise SystemExit(f"Pipeline could not complete: {exc}") from exc
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")
        run_batch(input_path, Path(args.output), retriever, client)


if __name__ == "__main__":
    main()
