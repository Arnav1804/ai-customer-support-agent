# Project log

## Session 1 — 2026-09-16

Set up project documentation and the environment-variable example. No support-agent metrics or model results were produced. Open issue: the data-preparation pipeline still needed to be implemented.

## Session 2 — 2026-09-16

Added standalone data-preparation scripts to inspect brand IDs, reconstruct full graph-connected brand conversations, and create a seed-42 sample of complete threads. Open issue: the scripts default to `data/raw/twcs.csv`; in this workspace the supplied source currently lives at `raw/twcs.csv`, so pass `--input raw/twcs.csv` or place it at the default path.

## Session 3 — 2026-09-16

Added a reading sampler that prints one random first customer message per thread without creating labels. Open issue: the golden-set builder will be added after the intent taxonomy is supplied; label columns will remain blank for manual annotation.

## Session 4 — 2026-09-16

Added a seed-7, 200-thread golden-set builder after receiving the ten-intent taxonomy. It balances customer-message length and position within a conversation, emits only the requested columns, and deliberately leaves every annotation field blank. Open issue: `golden_set.csv` is generated only after the filtered sample is available and must be manually labelled.

## Session 5 — 2026-09-16

Added a majority-intent/always-escalate baseline and a configurable regex/template baseline. The trivial baseline refuses to run before manual intent labels exist. The simple baseline uses an intentionally empty JSON keyword map awaiting user-supplied patterns, writes the requested four-column prediction schema, and stores generated replies separately to avoid changing that schema. Open issue: populate `baselines/intent_keywords.json` and hand-label `evaluation/golden_set.csv` before evaluating either baseline.

## Session 6 — 2026-09-16

Built the modular LLM pipeline: few-shot intent classification with self-rated confidence, scikit-learn TF-IDF retrieval over only the seeded sample, retrieval-grounded reply drafting, explicit escalation rules, and a single-message/batch CLI. LLM calls retry once; failed batch rows are retained with a warning and a skipped status. Decision: a thread ending in a brand turn is the operational proxy for a resolved precedent, and pipeline CSVs preserve the golden-set columns while adding explicit prediction columns rather than overwriting human labels. Open issue: install scikit-learn, configure `OPENAI_API_KEY` in `.env`, and run only after the processed sample exists.

## Session 7 — 2026-09-16

Added evaluation scripts for classification/escalation metrics, LLM reply judging, and LLM-versus-human agreement, plus a combined results table. Metrics exclude blank labels and failed/skipped predictions. Decision: macro F1 is the single intent-F1 figure in the results table while per-class F1 remains in `classification_metrics.json`; simple-template replies are judged with no historical evidence because that baseline does not retrieve precedents. Open issue: manually complete the golden labels and 30-row human judgement sheet before reporting any metrics.

## Session 8 — 2026-09-16

Rewrote the README with exact setup, data preparation, pipeline, and evaluation commands; added blank report and decision-log scaffolds. Decision: the README distinguishes a quick table refresh from the much longer manual-labelling and live-LLM workflow rather than promising an unsupported end-to-end time. Open issue: the user must fill the report, decision log, labels, keyword patterns, and human-judge sheet with real evidence before submission.
