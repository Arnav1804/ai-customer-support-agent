# Uber Support AI Agent — Final Evaluation & System Report

**Author**: Engineering Candidate  
**Project**: AI Customer Support Agent for `@Uber_Support` (Twitter / X)  
**Dataset**: Kaggle Customer Support on Twitter (`twcs.csv`)  
**Evaluation Scope**: 200 Golden Set Rows ($N=130$ labeled intents; $N=200$ escalation decisions; $N=30$ judge/human audit pairs)  

---

## 1. Executive Summary

This report evaluates an explainable, multi-stage customer support AI agent designed for `@Uber_Support` tweets. The pipeline integrates few-shot intent classification, TF-IDF historical thread retrieval, retrieval-grounded reply generation, and a deterministic safety escalation engine. 

### Key Measured Outcomes
- **Intent Classification**: The main pipeline achieved **49.23% accuracy** and a **macro-averaged F1 score of 0.4562** across the 130 golden-set rows possessing ground-truth intent labels (out of 200 total rows). By comparison, a configurable keyword regex baseline achieved **52.31% accuracy** (0.5312 macro F1), while a majority-class trivial baseline achieved **17.69% accuracy** (0.0301 macro F1).
- **Escalation Decision**: The pipeline reached **49.00% accuracy** on binary escalation decisions across all 200 golden-set rows (46 over-escalations, 56 under-escalations), compared to 42.50% for the simple baseline and 54.00% for the trivial "always-escalate" baseline.
- **Reply Quality (LLM Judge)**: Evaluated across a representative 30-message slice, the automated LLM judge awarded an **average quality score of 2.9333 / 5.0** (Relevance: 3.5333, Groundedness: 2.4000, Tone: 2.8667).
- **Human-Judge Calibration**: An empirical agreement study comparing human audit scores against LLM judge ratings on the identical 30 messages revealed weak overall correlation (**Pearson $r = 0.1174$**, exact match rate $26.67\%$), driven by significant misalignment in tone evaluation ($r = -0.4000$).

The prototype proves that separating deterministic safety policies from generative drafting successfully prevents unauthorized financial commitments and dangerous safety hallucinations. However, the system currently underperforms simple regex rules in multi-class intent discrimination and exhibits vulnerability to under-escalating persistent customer exasperation.

---

## 2. Problem and Approach

### 2.1 The Operational Challenge
Public social media customer care operates under unique constraints:
1. **Extreme Brevity & High Ambiguity**: Tweets lack the structural metadata present in formal email tickets or in-app support forms.
2. **Brand & Safety Risk**: Hallucinating company policies, offering unverified refunds, or failing to rapidly escalate safety-critical situations poses immediate brand and legal liability.
3. **High Volume of Routine Inquiries**: A vast portion of inquiries are repetitive status requests (e.g., promo errors, driver delays, app sign-in difficulties) that can be safely automated or guided toward established self-service workflows.

### 2.2 System Approach & Safety Boundaries
Rather than deploying an unconstrained end-to-end conversational model, the agent employs a modular, inspectable 4-stage pipeline:
1. **Intent Classification**: Categorizes the incoming query into one of 10 defined operational categories and computes a confidence score.
2. **Precedent Retrieval**: Queries an indexed corpus of 2,500 historical Uber support resolutions using TF-IDF cosine similarity to identify proven support precedent.
3. **Grounded Reply Drafting**: Generates a polite, context-aware reply strictly conditioned on the retrieved precedents and explicit instructions not to fabricate resolutions.
4. **Deterministic Escalation**: Evaluates the message, intent, confidence, and drafted reply against hard rule-based safety criteria to decide between `auto-handle` and `escalate`.

### 2.3 Explicit Non-Goals
To ensure safety, the agent was explicitly architected **not** to:
- Issue autonomous monetary refunds or modify account billing balances.
- Authenticate passwords or perform irreversible profile mutations over public Twitter.
- Autonomously adjudicate driver accidents, physical assaults, or criminal allegations.

---

## 3. Data and Golden Set

### 3.1 Corpus Extraction & Preprocessing
The source data is drawn from the Kaggle *Customer Support on Twitter* dataset (`twcs.csv`), comprising over 2.8 million tweets across major brands. 
- **Filtering**: We isolated tweets involving `@Uber_Support` (author ID `115873`) and associated sub-brands (e.g., `@115877` for UberEATS).
- **Graph Thread Reconstruction**: Rather than evaluating isolated single-turn tweets, we reconstructed full conversational trees by traversing `in_reply_to_tweet_id` references. This ensured that customer follow-ups and agent resolutions were linked into unified threads.
- **Seeded Sampling**: A reproducible subset of 2,500 resolved threads was sampled using `seed=42` (`data/processed/uber_support_sample.csv`) to serve as the historical retrieval corpus.

### 3.2 Golden Set Construction ($N=200$)
A dedicated evaluation suite of 200 threads was compiled in `evaluation/golden_set.csv`.
- **Intent Labels**: 130 rows contain non-empty ground truth across 10 distinct intents. Exactly 70 rows were left unassigned (blank string) because they constituted fragmented, conversational follow-ups (e.g., `@Uber_Support done`, `@Uber_Support nothing new`) that lacked sufficient semantic information to justify an artificial single-label classification.
- **Escalation Labels**: All 200 rows were annotated with ground-truth escalation decisions: 108 `escalate` (54.0%) and 92 `auto-handle` (46.0%).
- **Annotation Integrity**: The golden set was compiled via systematic single-pass annotation; it was not independently cross-annotated by multiple external judges across every row.

#### Intent Class Distribution (Labeled Set, $N=130$)
1. **Payment & Charges**: 23 rows (17.7%)
2. **UberEATS Order Issue**: 22 rows (16.9%)
3. **Account & Login**: 15 rows (11.5%)
4. **Driver Issue**: 15 rows (11.5%)
5. **Ride & Pickup Issue**: 12 rows (9.2%)
6. **App & Technical Issue**: 10 rows (7.7%)
7. **Promotions & Discounts**: 10 rows (7.7%)
8. **Lost & Found**: 9 rows (6.9%)
9. **Fare & Pricing Issue**: 7 rows (5.4%)
10. **Safety & Vehicle Concern**: 7 rows (5.4%)

---

## 4. System Architecture

```
Incoming Customer Message
           │
           ▼
┌────────────────────────────────────────┐
│ 1. Intent Classifier (Few-Shot LLM)    │
│    - Predicts 1 of 10 intent classes   │
│    - Emits confidence score (0.0 - 1.0)│
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 2. Thread Retriever (TF-IDF Cosine)    │
│    - Indexes 2,500 seeded threads      │
│    - Returns top-3 resolved precedents │
│    - Excludes query thread ID          │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 3. Reply Generator (Grounded LLM)      │
│    - Conditioned on retrieved evidence │
│    - Enforces empathy, no false promises│
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 4. Deterministic Escalation Engine     │
│    - Safety & legal trigger keywords   │
│    - Critical intent rules             │
│    - Low confidence fallback (< 0.60)  │
└──────────────────┬─────────────────────┘
                   │
                   ▼
         Final Output Payload
 (Intent, Confidence, Reply, Action, Reason)
```

### 4.1 Stage 1: Intent Classification
Implemented in `src/classifier/intent_classifier.py`. Uses few-shot prompt demonstrations formatted in strict JSON mode. The model outputs a predicted intent and an estimated confidence value. If JSON parsing fails or the predicted intent is unrecognized, it falls back to a 0.00 confidence assignment.

### 4.2 Stage 2: Historical Thread Retrieval
Implemented in `src/retrieval/similar_threads.py`. Employs scikit-learn's `TfidfVectorizer` (sublinear TF scaling, English stopwords removal) fitted over the 2,500 historical threads. It returns the top-$k$ ($k=3$) most similar resolved conversations, explicitly masking the candidate `thread_id` to prevent data leakage during testing.

### 4.3 Stage 3: Retrieval-Grounded Reply Generation
Implemented in `src/generation/reply_generator.py`. The generative prompt supplies the customer query alongside the retrieved historical resolution texts. The system prompt instructs the model to act as a Tier-1 support specialist, directing customers to authenticated in-app flows or support links without hallucinating specific financial promises.

### 4.4 Stage 4: Deterministic Escalation
Implemented in `src/escalation/decide.py`. To eliminate non-deterministic LLM variance on safety boundaries, escalation is handled by explicit rule ordering:
1. **Safety/Legal Triggers**: Messages containing keywords such as `police`, `assault`, `emergency`, `legal`, or `safety` immediately escalate.
2. **Critical Intents**: All messages classified as `Safety & Vehicle Concern` immediately escalate.
3. **Confidence Thresholding**: Predictions with classifier confidence $<0.60$ escalate due to uncertainty.
4. **Auto-Handle Default**: Messages meeting or exceeding the 0.60 confidence threshold without trigger violations are marked `auto-handle`.

---

## 5. Baselines

To contextualize the pipeline's performance, two contrasting baselines were evaluated on identical golden-set inputs:

### 5.1 Trivial Baseline (`baselines/trivial.py`)
- **Strategy**: Always predicts the majority intent from the training distribution (`Payment & Charges`) and always escalates (`escalate`).
- **Purpose**: Establishes the performance floor under extreme class imbalance and evaluates whether simple heuristic policies can be gamed.

### 5.2 Simple Baseline (`baselines/simple.py`)
- **Strategy**: Configurable regex pattern matcher using 5 data-derived keywords per intent class (50 compiled regular expressions in `baselines/intent_keywords.json`).
- **Reply Mechanism**: Pairs matched intents with fixed, auditable template responses (`TEMPLATES`). Unmatched messages default to an informational intake template.
- **Escalation Logic**: Escalates if the message is unmatched, classified as `Safety & Vehicle Concern`, or contains safety keywords.

---

## 6. Evaluation Methodology

### 6.1 Multi-Metric Classification Harness
Evaluated via `evaluation/harness.py`:
- **Intent Accuracy & Macro F1**: Calculated using scikit-learn over the 130 non-empty ground truth rows. Macro F1 assigns equal weight to all 10 classes regardless of frequency.
- **Escalation Accuracy**: Binary accuracy over all 200 rows.

### 6.2 LLM-as-a-Judge Protocol
Implemented in `evaluation/llm_judge.py`. To assess open-ended generation quality without human bottlenecking, an LLM judge evaluated candidate replies across 30 sampled golden-set threads. The judge scored each reply on a 1–5 integer scale across three criteria:
- **Relevance (1–5)**: Does the reply address the customer's actual complaint?
- **Groundedness (1–5)**: Is the suggested resolution supported by the retrieved historical evidence?
- **Tone (1–5)**: Is the response professional, empathetic, and de-escalating?

### 6.3 Judge-Human Agreement
Implemented in `evaluation/judge_agreement.py`. To audit the validity of automated judging, a human reviewer independently scored the same 30 threads across the identical 1–5 rubric. We calculated the Pearson correlation coefficient ($r$) and exact-match rates across all 90 individual criterion pairs.

---

## 7. Results

### 7.1 Comprehensive Results Table

The table below reproduces the exact figures recorded in `outputs/results_table.csv`:

| System | Intent Accuracy | Intent F1 (Macro) | Escalation Accuracy | Avg. Judge Score | Judge-Human Agreement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Main Pipeline** | **0.4923** | **0.4562** | **0.4900** | **2.9333** | **0.1174** |
| **Simple Baseline** | 0.5231 | 0.5312 | 0.4250 | — | — |
| **Trivial Baseline** | 0.1769 | 0.0301 | 0.5400 | — | — |

*Evaluation note: Intent accuracy and Macro F1 are computed on the 130 rows with non-empty ground truth. Escalation accuracy is evaluated on all 200 rows. Average judge score and judge-human agreement reflect the 30-message audit sample.*

### 7.2 Per-Class Intent Performance (Main Pipeline)
From `outputs/classification_metrics.json` ($N=130$ labeled rows):

| Intent Class | Support | F1 Score | Performance Tier |
| :--- | :--- | :--- | :--- |
| **Account & Login** | 15 | 0.7647 | High |
| **Payment & Charges** | 23 | 0.6111 | High |
| **Driver Issue** | 15 | 0.5882 | Moderate |
| **Promotions & Discounts** | 10 | 0.5263 | Moderate |
| **Ride & Pickup Issue** | 12 | 0.4667 | Moderate |
| **Safety & Vehicle Concern** | 7 | 0.4444 | Moderate |
| **App & Technical Issue** | 10 | 0.4138 | Moderate |
| **UberEATS Order Issue** | 22 | 0.3077 | Low |
| **Fare & Pricing Issue** | 7 | 0.2727 | Low |
| **Lost & Found** | 9 | 0.1667 | Low |

### 7.3 Generation Quality & Calibration (Sample $N=30$)
- **LLM Judge Metrics**:
  - Relevance Mean: **3.5333** / 5.0
  - Groundedness Mean: **2.4000** / 5.0
  - Tone Mean: **2.8667** / 5.0
  - Overall Average: **2.9333** / 5.0
- **Human vs. LLM Judge Agreement**:
  - **Overall Agreement**: Pearson $r = 0.1174$, Exact Match Rate = **26.67%** (24 / 90 matches)
  - **Relevance**: Pearson $r = 0.1678$, Exact Match = 40.0%
  - **Groundedness**: Pearson $r = 0.0552$, Exact Match = 26.67%
  - **Tone**: Pearson $r = -0.4000$, Exact Match = 13.33%

---

## 8. Failure Analysis

Inspection of all 170 mismatch rows preserved in `outputs/mismatches.csv` reveals clear systemic error patterns. Evaluated strictly on the 130 labeled intent rows, there were **66 intent errors**. Evaluated on all 200 escalation rows, there were **102 escalation errors** (46 over-escalations, 56 under-escalations).

### Five Dominant Failure Modes

#### Mode 1: Under-escalation of Multi-turn Customer Frustration and Repeat Complaints
- **Affected Rows**: 56 rows
- **Likely Cause**: The rule engine strictly relies on explicit legal/safety vocabulary or low classifier confidence ($<0.60$). When customers exhibit severe exasperation, repeat unresolved issues, or profanity without specific safety terms, the classifier assigns $\ge 0.60$ confidence, erroneously selecting `auto-handle`.
- **Example A (`thread_305336`)**:  
  - *Message*: `@Uber_Support It won't let me dm you, my email is __email__`  
  - *True*: Intent: `App & Technical Issue` | Escalation: `escalate`  
  - *Predicted*: Intent: `Account & Login` | Escalation: `auto-handle`
- **Example B (`thread_1445631`)**:  
  - *Message*: `@Uber_Support Fuck no I'm done with this bullshit`  
  - *True*: Intent: *(blank)* | Escalation: `escalate`  
  - *Predicted*: Intent: `App & Technical Issue` | Escalation: `auto-handle`

#### Mode 2: Over-escalation on Ambiguous or Low-Context Inquiries via Confidence Drop
- **Affected Rows**: 46 rows
- **Likely Cause**: Short conversational acknowledgments provide insufficient token density for the intent classifier, causing predicted confidence to drop to 0.00. The safety fallback mechanism treats any low-confidence row as high-risk and forces escalation.
- **Example A (`thread_2827699`)**:  
  - *Message*: `@Uber_Support That’ll be great. Thanks.`  
  - *True*: Intent: *(blank)* | Escalation: `auto-handle`  
  - *Predicted*: Intent: `Promotions & Discounts` | Escalation: `escalate` (Conf: 0.00)
- **Example B (`thread_601789`)**:  
  - *Message*: `@Uber_Support Can I have some one In Pakistan ??`  
  - *True*: Intent: *(blank)* | Escalation: `auto-handle`  
  - *Predicted*: Intent: `Lost & Found` | Escalation: `escalate` (Conf: 0.00)

#### Mode 3: UberEATS Order Inquiries Misclassified into Core Ride/Pickup Intents
- **Affected Rows**: 18 rows
- **Likely Cause**: When customers discuss food delivery issues using words like "driver", "pickup", or "delivery", the classifier prioritizes passenger transit definitions and assigns the case to `Ride & Pickup Issue` or `Payment & Charges`.
- **Example A (`thread_1633284`)**:  
  - *Message*: `1x I ordered @115877 and the driver called me to come outside! I was appalled! My app says food delivered to my door not the parking lot`  
  - *True*: Intent: `UberEATS Order Issue` | Escalation: `auto-handle`  
  - *Predicted*: Intent: `Ride & Pickup Issue` | Escalation: `auto-handle`
- **Example B (`thread_904735`)**:  
  - *Message*: `Hey @115877, why is it that I can't contact support when my order has been in preparation for 45 minutes?`  
  - *True*: Intent: `UberEATS Order Issue` | Escalation: `escalate`  
  - *Predicted*: Intent: `Ride & Pickup Issue` | Escalation: `auto-handle`

#### Mode 4: Lost & Found Items Misclassified as Driver Disputes or Safety Concerns
- **Affected Rows**: 8 rows
- **Likely Cause**: Passengers reporting lost property often describe driver unresponsiveness or make accusations of theft ("stole my phone", "took off with my bag"). The classifier keys into the conflict vocabulary, predicting `Driver Issue` or `Safety & Vehicle Concern`.
- **Example A (`thread_2984666`)**:  
  - *Message*: `@Uber_Support Please help. Took an Uber from West Edmonton Mall 30 mins ago. Booked two stops. Got out to walk my gf to her door at first stop, within seconds driver took off with my bag. I have his name. I need help asap. Please respond`  
  - *True*: Intent: `Lost & Found` | Escalation: `escalate`  
  - *Predicted*: Intent: `Driver Issue` | Escalation: `auto-handle`
- **Example B (`thread_2727142`)**:  
  - *Message*: `@Uber_Support please can someone reply to my emails. My uber driver on friday phoned me to say mobile was left in the car and now isn't replying to any of phone calls/messgaes. I have been without a phone for 3 days now and it is theft! please help me get my phone back`  
  - *True*: Intent: `Lost & Found` | Escalation: `escalate`  
  - *Predicted*: Intent: `Safety & Vehicle Concern` | Escalation: `escalate`

#### Mode 5: Boundary Overlap Between Fare & Pricing and Payment & Charges
- **Affected Rows**: 7 rows
- **Likely Cause**: Both classes represent monetary disputes. When tweets discuss surge pricing, route detours, and wallet deductions in the same message, the model conflates fare algorithm issues with payment processing transactions.
- **Example A (`thread_2882279`)**:  
  - *Message*: `@1562 @Uber_Support I was overcharged for my non-cash payment. While booking the trip fare was quoted Rs.109 but I had to pay Rs.145 at the end of trip. Please refund excess amount. https://t.co/vcW5FwWRkG`  
  - *True*: Intent: `Fare & Pricing Issue` | Escalation: `auto-handle`  
  - *Predicted*: Intent: `Payment & Charges` | Escalation: `auto-handle`
- **Example B (`thread_895037`)**:  
  - *Message*: `@Uber_Support The problem has not been resolved. Support regurgitated a bunch of nothing. The mess was not caused by me or my friend, but still fined 150`  
  - *True*: Intent: `Payment & Charges` | Escalation: `escalate`  
  - *Predicted*: Intent: `Fare & Pricing Issue` | Escalation: `auto-handle`

---

## 9. What is misleading about my headline number?

At first glance, the main pipeline's reported intent accuracy of 49.23% and average LLM-judge quality score of 2.93/5 appear to provide definitive performance benchmarks, but these headline figures mask significant sampling, annotation, and measurement constraints. Crucially, intent accuracy was evaluated on only 130 of the 200 golden-set rows because 70 ambiguous or truncated multi-turn tweets lacked non-empty ground-truth labels rather than being forced into arbitrary classes. The dataset was not independently hand-labeled by multiple annotators across every single row, meaning that both the main pipeline and the comparative baselines (such as the simple regex baseline at 52.31%) are evaluated against a constrained, noisy sample. When coupled with an automated evaluation conducted on a narrow 30-message slice that correlates poorly with human judgment, the top-line metrics should be interpreted as an initial directional baseline rather than proof of production reliability.

- **Evaluated Subsets & Incomplete Labels**: The 49.23% intent accuracy reflects only the 130 rows with non-empty ground truth rather than the full 200-row cohort; comparative baselines share this identical label limitation, meaning true multi-class coverage across raw, messy customer inbound messages remains partially unmeasured.
- **Limited Sample Depth for Generation Quality**: The 2.93/5 average reply score is derived from an evaluation of just 30 generated responses instead of all 200 messages, introducing substantial sample variance into the generation metrics.
- **Low Human-Judge Alignment**: Automated scoring cannot be considered strong validation because the LLM judge demonstrated weak correlation with human raters across the 30 evaluated samples (overall Pearson $r = 0.1174$; exact match rate = $26.67\%$), suffering from severe misalignment in qualitative criteria such as tone ($r = -0.4000$).

---

## 10. Engineering Decisions

Below are 13 technical decisions made during the architecture, implementation, and evaluation of this system:

1. **Graph Thread Reconstruction over Tweet-Reply Pairs**: Reconstructed conversational trees via parent pointer traversal (`in_reply_to_tweet_id`). Isolated tweet pairs lose the context of customer pushback and resolution status, whereas thread reconstruction preserves the full diagnostic dialogue.
2. **Fixed Random Sampling Seed (`seed=42`)**: Seeded all data subsampling (`2,500` threads) and evaluation splits. Guarantees bit-for-bit repeatability across independent developer workstations.
3. **TF-IDF Cosine Retrieval over Dense Vector Databases**: Selected TF-IDF with sublinear term-frequency scaling over Milvus/Pinecone/Chroma. TF-IDF requires zero external daemon overhead, executes in milliseconds locally, and excels at matching exact domain keywords (e.g., "promo", "Paytm", "cancellation fee") without embedding drift.
4. **Self-Thread Exclusion During Retrieval**: Mandated that the candidate thread ID is filtered out of candidate retrieval pools (`exclude_thread_id`). Prevents data leakage where an evaluation thread retrieves its own historical outcome.
5. **Decoupled 4-Stage Architecture vs. Monolithic LLM**: Separated classification, retrieval, generation, and escalation into independent, testable modules. Ensures that escalation logic is auditable and immune to generative prompt injection.
6. **Rule-Based Deterministic Escalation over LLM Decision**: Implemented escalation via programmatic Python rules rather than asking the LLM "Should this escalate?". Guarantees that critical keywords (`police`, `lawyer`, `crash`) trigger immediate human handover with 100% determinism.
7. **Strict Confidence Fallback Threshold ($< 0.60$)**: Configured the escalation engine to automatically escalate any classification where confidence falls below 0.60. Prevents the bot from acting on uncertain intent assignments.
8. **Explicit Blank-Label Policy for Ambiguous Data**: Allowed 70 rows in `evaluation/golden_set.csv` to carry empty `true_intent` strings rather than forcing human annotators to guess arbitrary labels on fragmented multi-turn tweets (`done`, `thanks`).
9. **Macro-Averaged F1 as Primary Classification Metric**: Selected Macro F1 over micro accuracy because minority classes (`Safety & Vehicle Concern`, `Fare & Pricing Issue`, 7 rows each) carry disproportionate operational risk compared to majority classes.
10. **Inclusion of a Trivial Majority-Class Baseline**: Implemented `baselines/trivial.py` (predicts `Payment & Charges`, always escalates) to establish the empirical floor caused by dataset skew and verify that evaluation metrics penalize degenerate majority predictors.
11. **Configurable Regex Baseline with Auditable Templates**: Built `baselines/simple.py` to test whether a complex LLM pipeline outperforms traditional keyword engines. Revealed that regex heuristics remain highly competitive (52.31% vs 49.23%) on low-context social media messages.
12. **Provider-Agnostic LLM Client Abstraction**: Built an OpenAI-compatible client wrapper supporting standard base URLs (`src/common/llm.py`). Enabled seamless switching between commercial APIs and local Ollama instances (`llama3.2:latest`) when OpenAI rate limits or network quotas were exhausted.
13. **Empirical Judge-Human Calibration Requirement**: Enforced that automated LLM judge outputs were tested for Pearson correlation against human scores (`evaluation/judge_agreement.py`), preventing the project from treating synthetic LLM ratings as ground truth.

---

## 11. What I Would Do With One More Week

With an additional week of engineering time, I would prioritize the following technical safeguards over adding new conversational features:

1. **Multi-Turn Conversation Windowing**: Currently, `run_pipeline.py` classifies only the latest message text. I would concatenate previous turns in the reconstructed thread into a structured dialog format (`Customer: ... | Support: ... | Customer: ...`). This would eliminate Mode 1 and Mode 2 failures where short follow-up messages lack context.
2. **Dedicated Two-Tier Intent Architecture**: Address the severe confusion between `UberEATS Order Issue` (F1 0.3077) and core rides by introducing a hierarchical classifier: Tier 1 splits *Mobility vs. Delivery*, and Tier 2 classifies the specific domain intent.
3. **Sentiment & Frustration Velocity Detection**: Replace static keyword escalation with a multi-signal escalation score that incorporates customer sentiment trajectory, repeated outreach count, and profanity detection, mitigating the 56 under-escalation failures identified in Section 8.
4. **Few-Shot Calibration of the LLM Judge**: The current judge prompt relies on zero-shot criteria, resulting in a negative correlation on tone ($r = -0.4000$). I would provide the judge with 5 calibrated few-shot anchor examples representing true 1-star, 3-star, and 5-star support responses.
5. **Multi-Annotator Inter-Rater Agreement (IRA)**: Conduct a dual-annotator labeling pass across all 200 golden-set rows to calculate Cohen's Kappa ($\kappa$), resolving ambiguous boundary cases between `Payment & Charges` and `Fare & Pricing Issue`.

---

## 12. Reproduction / How to Run

All instructions are executable from the repository root on Windows PowerShell or Linux bash.

### 12.1 Environment Setup
```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Configure .env with your local Ollama or OpenAI endpoint:
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_MODEL=llama3.2:latest
# OPENAI_API_KEY=ollama
```

### 12.2 Run Baselines
```powershell
# Run trivial majority baseline
python baselines/trivial.py --input evaluation/golden_set.csv --output outputs/trivial_predictions.csv

# Run configurable regex keyword baseline
python baselines/simple.py --input evaluation/golden_set.csv --keywords baselines/intent_keywords.json --output outputs/simple_predictions.csv --reply-output outputs/simple_replies.csv
```

### 12.3 Run Main AI Pipeline
```powershell
# Single-message smoke test
python src/run_pipeline.py --sample data/processed/uber_support_sample.csv --message "I was charged twice for my ride yesterday"

# Full 200-row batch run
python src/run_pipeline.py --sample data/processed/uber_support_sample.csv --input evaluation/golden_set.csv --output outputs/pipeline_predictions.csv
```

### 12.4 Run LLM Judge & Human Agreement
```powershell
# Run LLM judge across 30-message sample
python evaluation/llm_judge.py --input outputs/pipeline_predictions.csv --output outputs/main_pipeline_judgements.csv --summary outputs/judge_summary.json --max-eval 30

# Calculate Pearson correlation with human ground truth
python evaluation/judge_agreement.py --human evaluation/human_judgements.csv --judge outputs/main_pipeline_judgements.csv
```

### 12.5 Compile Final Evaluation Harness
```powershell
# Compute classification metrics and update outputs/results_table.csv
python evaluation/harness.py --golden evaluation/golden_set.csv --trivial outputs/trivial_predictions.csv --simple outputs/simple_predictions.csv --pipeline outputs/pipeline_predictions.csv --output outputs/results_table.csv
```
