"""Compute label-backed classification metrics and assemble the final results table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


SYSTEMS = {
    "main pipeline": "outputs/pipeline_predictions.csv",
    "simple baseline": "outputs/simple_predictions.csv",
    "trivial baseline": "outputs/trivial_predictions.csv",
}
RESULT_COLUMNS = [
    "system", "intent_accuracy", "intent_f1_macro", "escalation_accuracy",
    "avg_judge_score", "judge_human_agreement_main_pipeline_only",
]


def read_csv_by_thread(path: str | Path) -> dict[str, dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        thread_id = (row.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError(f"Missing thread_id in {csv_path}")
        if thread_id in indexed:
            raise ValueError(f"Duplicate thread_id '{thread_id}' in {csv_path}")
        indexed[thread_id] = row
    return indexed


def normalise_escalation(value: str | None) -> str | None:
    normalized = (value or "").strip().casefold().replace("_", "-")
    if normalized in {"escalate", "yes", "true", "1", "escalated"}:
        return "escalate"
    if normalized in {"auto-handle", "autohandle", "no", "false", "0", "do not escalate"}:
        return "auto-handle"
    return None


def score_system(golden: dict[str, dict[str, str]], predictions: dict[str, dict[str, str]]) -> dict:
    intent_pairs = []
    escalation_pairs = []
    for thread_id, gold in golden.items():
        prediction = predictions.get(thread_id)
        if not prediction:
            continue
        true_intent = (gold.get("true_intent") or "").strip()
        predicted_intent = (prediction.get("predicted_intent") or "").strip()
        if true_intent and predicted_intent:
            intent_pairs.append((true_intent, predicted_intent))
        true_escalation = normalise_escalation(gold.get("should_escalate"))
        predicted_escalation = normalise_escalation(prediction.get("predicted_escalate"))
        if true_escalation and predicted_escalation:
            escalation_pairs.append((true_escalation, predicted_escalation))

    metrics: dict = {
        "intent_evaluated_rows": len(intent_pairs),
        "escalation_evaluated_rows": len(escalation_pairs),
        "intent_accuracy": None,
        "intent_f1_macro": None,
        "intent_f1_per_class": {},
        "escalation_accuracy": None,
    }
    if intent_pairs:
        y_true, y_pred = zip(*intent_pairs)
        labels = sorted(set(y_true), key=str.casefold)
        _, _, f1_values, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        metrics["intent_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["intent_f1_macro"] = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        metrics["intent_f1_per_class"] = {
            label: {"f1": float(f1), "support": int(count)}
            for label, f1, count in zip(labels, f1_values, support)
        }
    if escalation_pairs:
        y_true, y_pred = zip(*escalation_pairs)
        metrics["escalation_accuracy"] = accuracy_score(y_true, y_pred)
    return metrics


def load_json_if_present(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc


def _csv_value(value: float | int | None) -> str:
    return "" if value is None else f"{float(value):.4f}"


def write_results_table(metrics: dict, judge_summary: dict, agreement: dict, output_path: str | Path) -> None:
    """Write blanks for unavailable judge/human values; never manufacture a score."""
    summary_by_system = judge_summary.get("systems", {})
    main_agreement = agreement.get("overall", {}).get("pearson_correlation")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for system in SYSTEMS:
            classification = metrics.get("systems", {}).get(system, {})
            writer.writerow({
                "system": system,
                "intent_accuracy": _csv_value(classification.get("intent_accuracy")),
                "intent_f1_macro": _csv_value(classification.get("intent_f1_macro")),
                "escalation_accuracy": _csv_value(classification.get("escalation_accuracy")),
                "avg_judge_score": _csv_value(summary_by_system.get(system, {}).get("average_judge_score")),
                "judge_human_agreement_main_pipeline_only": _csv_value(main_agreement) if system == "main pipeline" else "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline and baseline predictions against manual labels.")
    parser.add_argument("--golden", default="evaluation/golden_set.csv")
    parser.add_argument("--pipeline", default=SYSTEMS["main pipeline"])
    parser.add_argument("--simple", default=SYSTEMS["simple baseline"])
    parser.add_argument("--trivial", default=SYSTEMS["trivial baseline"])
    parser.add_argument("--metrics-output", default="outputs/classification_metrics.json")
    parser.add_argument("--judge-summary", default="outputs/judge_summary.json")
    parser.add_argument("--agreement", default="outputs/judge_agreement.json")
    parser.add_argument("--results-output", default="outputs/results_table.csv")
    args = parser.parse_args()

    golden = read_csv_by_thread(args.golden)
    if not golden:
        raise SystemExit(f"No golden-set rows found in {args.golden}.")
    paths = {"main pipeline": args.pipeline, "simple baseline": args.simple, "trivial baseline": args.trivial}
    report = {"golden_rows": len(golden), "systems": {}}
    for system, path in paths.items():
        predictions = read_csv_by_thread(path)
        report["systems"][system] = score_system(golden, predictions)
        print(
            f"{system}: {len(predictions):,} prediction rows; "
            f"intent n={report['systems'][system]['intent_evaluated_rows']:,}; "
            f"escalation n={report['systems'][system]['escalation_evaluated_rows']:,}"
        )

    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_results_table(report, load_json_if_present(args.judge_summary), load_json_if_present(args.agreement), args.results_output)
    print(f"Wrote classification metrics to {metrics_output}")
    print(f"Wrote combined results table to {args.results_output}")


if __name__ == "__main__":
    main()
