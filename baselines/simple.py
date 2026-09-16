"""Configurable regex baseline with fixed, auditable reply templates."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


OUTPUT_COLUMNS = ["thread_id", "predicted_intent", "predicted_escalate", "predicted_reason"]
REPLY_COLUMNS = ["thread_id", "predicted_intent", "predicted_reply"]
TEMPLATES = {
    "Fare & Pricing Issue": "I’m sorry the fare or pricing was unexpected. Please review the trip receipt in the Uber app and contact support from that trip if you need help.",
    "Payment & Charges": "I’m sorry about the charge. Please review the trip receipt and payment method in the Uber app, then contact support from the trip for account-specific help.",
    "Account & Login": "I’m sorry you’re having trouble accessing your account. Please use the account-help options in the Uber app so the support team can safely review it.",
    "App & Technical Issue": "I’m sorry the app is not working as expected. Please update the app and try again; if the problem continues, contact in-app support with the error details.",
    "Driver Issue": "I’m sorry about your experience with the driver. Please use the trip in the Uber app to share the details so the support team can review it.",
    "Safety & Vehicle Concern": "Your safety matters. Please use the trip’s in-app safety or help option to report the details so the safety team can review it promptly.",
    "Ride & Pickup Issue": "I’m sorry there was a problem with the ride or pickup. Please review the trip in the Uber app and contact support from it with the relevant details.",
    "Lost & Found": "I’m sorry you are missing an item. Please use the trip in the Uber app to start the lost-item process and contact the driver safely.",
    "UberEATS Order Issue": "I’m sorry there was an issue with your Uber Eats order. Please open the order in the app and use Help to report the problem.",
    "Promotions & Discounts": "I’m sorry the promotion or discount did not apply as expected. Please check the promotion terms in the Uber app and contact support if it still looks incorrect.",
    "Unmatched": "I’m sorry you’re having trouble. A support specialist will review your message and help with the next step.",
}


def load_patterns(path: Path) -> list[tuple[str, list[re.Pattern[str]]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read keyword config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("Keyword config must be a JSON object mapping intent names to regex lists.")
    compiled: list[tuple[str, list[re.Pattern[str]]]] = []
    for intent, patterns in raw.items():
        if intent not in TEMPLATES:
            raise SystemExit(f"No fixed reply template exists for configured intent: {intent}")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            raise SystemExit(f"Patterns for '{intent}' must be a JSON list of strings.")
        try:
            compiled.append((intent, [re.compile(pattern, re.IGNORECASE) for pattern in patterns]))
        except re.error as exc:
            raise SystemExit(f"Invalid regex for '{intent}': {exc}") from exc
    return compiled


def classify(text: str, configured_patterns: list[tuple[str, list[re.Pattern[str]]]]) -> tuple[str | None, list[str]]:
    """Choose the intent with the most matched configured patterns; ties use config order."""
    best_intent: str | None = None
    best_matches: list[str] = []
    for intent, patterns in configured_patterns:
        matches = [pattern.pattern for pattern in patterns if pattern.search(text)]
        if len(matches) > len(best_matches):
            best_intent, best_matches = intent, matches
    return best_intent, best_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a keyword/regex support-intent baseline.")
    parser.add_argument("--input", default="evaluation/golden_set.csv")
    parser.add_argument("--keywords", default="baselines/intent_keywords.json")
    parser.add_argument("--output", default="outputs/simple_predictions.csv")
    parser.add_argument("--reply-output", default="outputs/simple_replies.csv",
                        help="Separate file keeps generated replies out of the evaluation prediction schema")
    args = parser.parse_args()
    input_path, keyword_path = Path(args.input), Path(args.keywords)
    output_path, reply_output_path = Path(args.output), Path(args.reply_output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Build the golden set first.")
    if not keyword_path.exists():
        raise SystemExit(f"Keyword config not found: {keyword_path}. Add your regex lists first.")

    configured_patterns = load_patterns(keyword_path)
    active_patterns = sum(len(patterns) for _, patterns in configured_patterns)
    if not active_patterns:
        print(f"Warning: {keyword_path} contains no patterns yet; every message will be escalated.")
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {input_path}.")

    predictions, replies = [], []
    for row in rows:
        intent, matches = classify(row["message_text"], configured_patterns)
        if intent is None:
            predicted_intent = "Unmatched"
            predicted_escalate = "escalate"
            predicted_reason = "No configured keyword or regex matched; simple rule requires escalation."
        else:
            predicted_intent = intent
            predicted_escalate = "auto_handle"
            predicted_reason = "Matched configured pattern(s): " + ", ".join(matches)
        predictions.append({
            "thread_id": row["thread_id"], "predicted_intent": predicted_intent,
            "predicted_escalate": predicted_escalate, "predicted_reason": predicted_reason,
        })
        replies.append({
            "thread_id": row["thread_id"], "predicted_intent": predicted_intent,
            "predicted_reply": TEMPLATES[predicted_intent],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reply_output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader(); writer.writerows(predictions)
    with reply_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLY_COLUMNS)
        writer.writeheader(); writer.writerows(replies)

    intent_counts = Counter(item["predicted_intent"] for item in predictions)
    escalation_counts = Counter(item["predicted_escalate"] for item in predictions)
    print(f"Read {len(rows):,} golden-set rows using {active_patterns:,} configured regex patterns.")
    print("Predicted intent distribution: " + "; ".join(f"{intent}: {count:,}" for intent, count in sorted(intent_counts.items())))
    print("Predicted escalation distribution: " + "; ".join(f"{action}: {count:,}" for action, count in sorted(escalation_counts.items())))
    print(f"Wrote predictions to {output_path}")
    print(f"Wrote fixed-template replies to {reply_output_path}")


if __name__ == "__main__":
    main()
