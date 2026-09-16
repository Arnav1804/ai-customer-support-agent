# Uber Support Agent

An explainable support-agent prototype for `Uber_Support` tweets from Kaggle's [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). It classifies a customer message, retrieves comparable completed threads from a seeded Uber-only sample, drafts a retrieval-grounded reply, and applies explicit auto-handle/escalate rules.

## Prerequisites

- Python 3.10 or newer.
- A Kaggle account to download the source dataset.
- An OpenAI-compatible API key only for the LLM pipeline and LLM judge.

Run all commands below from the repository root:

```powershell
cd "C:\Users\yadua\OneDrive\Documents\Ai Ass\AI Agent For Cs"
```

## Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` locally. Do not commit `.env`.

```text
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

## Download and prepare data

Download **Customer Support on Twitter** from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), extract it, and place the CSV at exactly `data/raw/twcs.csv`.

If `twcs.csv` is already downloaded to your Downloads folder, use:

```powershell
New-Item -ItemType Directory -Force data\raw, data\processed, outputs | Out-Null
Copy-Item "$HOME\Downloads\twcs.csv" data\raw\twcs.csv
```

Confirm the support-account ID, reconstruct conversations, then take the reproducible seed-42 sample:

```powershell
python src\data_prep\inspect_brands.py --input data\raw\twcs.csv --brand uber
python src\data_prep\filter_brand.py --input data\raw\twcs.csv --author-id Uber_Support --output data\processed\uber_support.csv
python src\data_prep\subsample.py --input data\processed\uber_support.csv --output data\processed\uber_support_sample.csv --threads 2500 --seed 42
```

The extraction scripts read the Kaggle source only to create the processed data. All modelling, retrieval, pipeline, and evaluation commands below use only `data/processed/uber_support_sample.csv`.

## Create and hand-label the golden set

Print a reading sample if useful:

```powershell
python src\data_prep\sample_for_reading.py --input data\processed\uber_support_sample.csv --count 50 --seed 42
```

Create the 200-thread seed-7 golden-set template:

```powershell
python evaluation\build_golden_set.py --input data\processed\uber_support_sample.csv --output evaluation\golden_set.csv --threads 200 --seed 7
```

Hand-fill only these columns in `evaluation/golden_set.csv`:

```text
true_intent,should_escalate,escalate_reason,good_reply_notes
```

Do not treat blank labels as model results. The evaluation scripts exclude them.

## Run the systems

First add your own regexes to `baselines/intent_keywords.json`. The file intentionally contains empty arrays, so no keyword labels are invented by this repository.

Run the two baselines:

```powershell
python baselines\trivial.py --input evaluation\golden_set.csv --output outputs\trivial_predictions.csv
python baselines\simple.py --input evaluation\golden_set.csv --keywords baselines\intent_keywords.json --output outputs\simple_predictions.csv --reply-output outputs\simple_replies.csv
```

Run one message through the complete pipeline:

```powershell
python src\run_pipeline.py --sample data\processed\uber_support_sample.csv --message "I was charged twice for my trip"
```

Run the complete pipeline over the hand-labelled golden set:

```powershell
python src\run_pipeline.py --sample data\processed\uber_support_sample.csv --input evaluation\golden_set.csv --output outputs\pipeline_predictions.csv
```

Each LLM request retries once. In batch mode, any row that still fails is written with `pipeline_status` beginning `skipped:` and the batch continues.

## Evaluate and build the results table

Compute the classification and escalation metrics. This writes `outputs/classification_metrics.json` and an initial `outputs/results_table.csv`.

```powershell
python evaluation\harness.py --golden evaluation\golden_set.csv --pipeline outputs\pipeline_predictions.csv --simple outputs\simple_predictions.csv --trivial outputs\trivial_predictions.csv
```

Run the LLM judge for generated replies:

```powershell
python evaluation\llm_judge.py --golden evaluation\golden_set.csv --pipeline outputs\pipeline_predictions.csv --simple-replies outputs\simple_replies.csv --sample data\processed\uber_support_sample.csv
```

Create the 30-reply human-scoring sheet, score the three blank 1–5 columns yourself, then compare it with the LLM judge. The last command refreshes `outputs/results_table.csv`.

```powershell
python evaluation\judge_agreement.py --make-template --judge outputs\main_pipeline_judgements.csv --human evaluation\human_judgements.csv --count 30 --seed 7
# Fill relevance, groundedness, and tone in evaluation\human_judgements.csv.
python evaluation\judge_agreement.py --human evaluation\human_judgements.csv --judge outputs\main_pipeline_judgements.csv
```

`outputs/results_table.csv` has one row each for `main pipeline`, `simple baseline`, and `trivial baseline`. It reports intent accuracy, macro intent F1, escalation accuracy, judge score, and main-pipeline judge/human agreement. Unavailable values are blank, never invented.

## Reproduce the headline table in under 15 minutes

This is a **table-refresh path**, not a claim that fresh human labelling or hundreds of live LLM calls take under 15 minutes. With these already completed locally—hand-labelled `evaluation/golden_set.csv`, completed prediction files, `outputs/judge_summary.json`, and `evaluation/human_judgements.csv`—run:

```powershell
python evaluation\harness.py
python evaluation\judge_agreement.py
Get-Content outputs\results_table.csv
```

The full end-to-end process includes manual annotation and API-dependent calls, so its duration cannot be truthfully guaranteed. Keep the inputs and outputs used for a reported headline number together so it can be recomputed without changing labels or prompts.

## Project layout

```text
src/data_prep/       source inspection, graph thread reconstruction, sampling
src/classifier/      few-shot LLM intent classifier
src/retrieval/       TF-IDF cosine retrieval over the seeded sample only
src/generation/      retrieval-grounded LLM reply generator
src/escalation/      explicit escalation rules
baselines/           majority and regex/template baselines
evaluation/          golden set, metrics, LLM judge, human agreement
outputs/             local prediction and result artifacts
```
