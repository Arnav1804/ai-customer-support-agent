"""Compare 1-5 human reply scores with the LLM judge on the same thread IDs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

try:  # Supports both `python evaluation/judge_agreement.py` and module imports.
    from evaluation.harness import load_json_if_present, write_results_table
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    from harness import load_json_if_present, write_results_table


CRITERIA = ("relevance", "groundedness", "tone")


def read_rows(path: str | Path) -> dict[str, dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        thread_id = (row.get("thread_id") or "").strip()
        if thread_id:
            output[thread_id] = row
    return output


def score(value: str | None) -> int | None:
    try:
        parsed = int((value or "").strip())
    except ValueError:
        return None
    return parsed if 1 <= parsed <= 5 else None


def pearson(left: list[int], right: list[int]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else None


def compare(human: dict[str, dict[str, str]], judge: dict[str, dict[str, str]]) -> dict:
    pairs_by_criterion: dict[str, list[tuple[int, int]]] = {criterion: [] for criterion in CRITERIA}
    for thread_id, human_row in human.items():
        judge_row = judge.get(thread_id)
        if not judge_row or judge_row.get("judge_status") != "completed":
            continue
        for criterion in CRITERIA:
            human_score, judge_score = score(human_row.get(criterion)), score(judge_row.get(criterion))
            if human_score is not None and judge_score is not None:
                pairs_by_criterion[criterion].append((human_score, judge_score))

    per_criterion = {}
    flattened_human: list[int] = []
    flattened_judge: list[int] = []
    for criterion, pairs in pairs_by_criterion.items():
        human_scores = [pair[0] for pair in pairs]
        judge_scores = [pair[1] for pair in pairs]
        flattened_human.extend(human_scores)
        flattened_judge.extend(judge_scores)
        per_criterion[criterion] = {
            "comparable_scores": len(pairs),
            "pearson_correlation": pearson(human_scores, judge_scores),
            "exact_match_rate": sum(left == right for left, right in pairs) / len(pairs) if pairs else None,
        }
    return {
        "human_rows": len(human),
        "per_criterion": per_criterion,
        "overall": {
            "comparable_scores": len(flattened_human),
            "pearson_correlation": pearson(flattened_human, flattened_judge),
            "exact_match_rate": (
                sum(left == right for left, right in zip(flattened_human, flattened_judge)) / len(flattened_human)
                if flattened_human else None
            ),
        },
    }


def write_human_template(judge_rows: dict[str, dict[str, str]], output: Path, count: int, seed: int) -> None:
    eligible = [row for row in judge_rows.values() if row.get("judge_status") == "completed"]
    selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["thread_id", "message_text", "reply_text", "retrieved_thread_ids", *CRITERIA]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "thread_id": row["thread_id"], "message_text": row.get("message_text", ""),
                "reply_text": row.get("reply_text", ""), "retrieved_thread_ids": row.get("retrieved_thread_ids", ""),
                "relevance": "", "groundedness": "", "tone": "",
            })
    print(f"Wrote {len(selected):,} blank human-scoring rows to {output} (seed={seed}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report LLM-judge versus human agreement for main-pipeline replies.")
    parser.add_argument("--human", default="evaluation/human_judgements.csv",
                        help="Manual CSV with thread_id plus relevance, groundedness, and tone scores (1-5)")
    parser.add_argument("--judge", default="outputs/main_pipeline_judgements.csv")
    parser.add_argument("--predictions", default="outputs/pipeline_predictions.csv")
    parser.add_argument("--sample", default="data/processed/uber_support_sample.csv")
    parser.add_argument("--output", default="outputs/judge_agreement.json")
    parser.add_argument("--make-template", nargs="?", const="evaluation/human_judgements.csv", default=None,
                        help="Create a blank 30-reply human scoring sheet and exit")
    parser.add_argument("--count", "--max-rows", dest="count", type=int, default=30, help="Template rows when --make-template is used")
    parser.add_argument("--seed", type=int, default=7, help="Template sampling seed")
    parser.add_argument("--metrics", default="outputs/classification_metrics.json")
    parser.add_argument("--judge-summary", default="outputs/judge_summary.json")
    parser.add_argument("--results-output", default="outputs/results_table.csv")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    judge_path = Path(args.judge)
    if not judge_path.exists() and Path("outputs/pipeline_judgements.csv").exists():
        judge_path = Path("outputs/pipeline_judgements.csv")

    judge: dict[str, dict[str, str]] = {}
    if judge_path.exists():
        judge = read_rows(judge_path)

    human_path = Path(args.make_template) if (args.make_template and isinstance(args.make_template, str) and args.make_template != "evaluation/human_judgements.csv") else Path(args.human)
    if args.make_template is not None:
        if not judge and Path(args.predictions).exists():
            preds = read_rows(Path(args.predictions))
            judge = {
                tid: {
                    "thread_id": tid,
                    "message_text": r.get("message_text", ""),
                    "reply_text": r.get("drafted_reply", ""),
                    "retrieved_thread_ids": r.get("retrieved_thread_ids", ""),
                    "judge_status": "completed",
                }
                for tid, r in preds.items()
            }
        if not judge:
            raise SystemExit(f"Judge file not found: {judge_path}. Run llm_judge.py first.")
        write_human_template(judge, human_path, args.count, args.seed)
        return

    if not judge:
        raise SystemExit(f"Judge file not found: {judge_path}. Run llm_judge.py first.")
    try:
        human = read_rows(human_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Human-score file not found: {exc}. Run with --make-template, score its 1-5 columns, then rerun."
        ) from exc

    report = compare(human, judge)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    overall = report["overall"]
    print(f"Compared {overall['comparable_scores']:,} criterion scores from {report['human_rows']:,} human rows.")
    print(f"Overall Pearson correlation: {overall['pearson_correlation']}")
    print(f"Overall exact-match rate: {overall['exact_match_rate']}")
    print(f"Wrote agreement report to {output_path}")

    # Refresh the requested final table if the earlier evaluation artifacts exist.
    write_results_table(
        load_json_if_present(args.metrics), load_json_if_present(args.judge_summary), report, args.results_output
    )
    print(f"Refreshed combined results table at {args.results_output}")


if __name__ == "__main__":
    main()
