# Engineering Decisions Log

This document records 13 non-obvious engineering decisions made throughout the design, implementation, and evaluation of the Uber Support AI agent prototype. Each entry outlines the technical choice, rationale, trade-offs, and concrete implementation evidence in the repository.

---

### 1. Few-Shot In-Context Intent Classification with JSON Schema
- **Decision**: Implemented intent classification as a prompted few-shot task producing a strict JSON payload (`{"intent": "...", "confidence": float}`) constrained to 10 predefined operational categories, rather than training a custom BERT or fine-tuned classifier.
- **Why we chose it**: Allowed rapid zero-infrastructure iteration on intent taxonomy definitions, flexible prompt adjustments, and native probability estimation without managing training pipelines, model weights, or specialized GPU serving.
- **Trade-off / downside**: Incurs higher token latency per classification (~1–2 seconds per inference) and is vulnerable to occasional non-compliant JSON outputs or hallucinations when messages are ambiguous.
- **Evidence in current implementation**: [`src/classifier/intent_classifier.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/classifier/intent_classifier.py#L22-L72), where `INTENT_DEFINITIONS` and few-shot pairs are structured into a system prompt enforcing JSON response formatting.

---

### 2. Conservative Numeric Confidence Parsing with 0.00 Uncertainty Fallback
- **Decision**: Extracted confidence as an explicit float bounded strictly between 0.0 and 1.0, defaulting immediately to `0.00` if parsing fails, if the intent is unrecognized, or if schema validation fails.
- **Why we chose it**: Ensured that any generative failure, unparseable response, or malformed JSON is treated as maximum uncertainty, preventing the downstream pipeline from acting on unvalidated outputs.
- **Trade-off / downside**: Over-penalizes minor syntax anomalies (such as trailing text or unescaped quotes) by forcing confidence to zero, which subsequently triggers defensive human escalation.
- **Evidence in current implementation**: [`src/classifier/intent_classifier.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/classifier/intent_classifier.py#L42-L52), where invalid schema keys or unparseable numbers return `IntentPrediction(intent="Unassigned", confidence=0.0)`.

---

### 3. Sparse TF-IDF Cosine Retrieval over Dense Vector Databases
- **Decision**: Used an in-memory scikit-learn `TfidfVectorizer` with sublinear term-frequency scaling over 2,500 historical threads instead of deploying a dense vector store (e.g., Pinecone, Chroma, or embedding bi-encoders).
- **Why we chose it**: Customer support tweets rely heavily on exact domain tokens (e.g., "Paytm", "flat fare", "cancellation fee", "McDelivery"), which TF-IDF matches with microsecond latency and zero external service overhead or embedding costs.
- **Trade-off / downside**: Cannot capture semantic synonyms or conceptual paraphrasing that lack literal lexical overlap (e.g., failing to connect "driver took off" with "missing item").
- **Evidence in current implementation**: [`src/retrieval/similar_threads.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/retrieval/similar_threads.py#L25-L35), which fits a 10,000-feature `TfidfVectorizer(sublinear_tf=True, stop_words="english")` directly in Python.

---

### 4. Graph-Connected Thread Reconstruction as Resolution Proxy
- **Decision**: Reconstructed complete conversational trees via parent pointer traversal (`in_reply_to_tweet_id`) rather than treating single customer tweets and isolated replies as disconnected pairs.
- **Why we chose it**: A single support reply often only asks for customer details; full threads are necessary to verify whether a resolution link was sent, whether the customer pushed back, or whether the ticket concluded.
- **Trade-off / downside**: Increases preprocessing complexity, requires resolving circular or broken Twitter reply references, and introduces variable-length text into the retrieval index.
- **Evidence in current implementation**: [`src/data_prep/filter_brand.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/data_prep/filter_brand.py#L22-L78), which traces tweet parent-child graphs to reconstruct full dialogue threads.

---

### 5. Mandatory Self-Thread Exclusion During Retrieval
- **Decision**: Explicitly filtered the query message's own `thread_id` out of the candidate retrieval pool (`exclude_thread_id`).
- **Why we chose it**: Prevented evaluation data leakage where an incoming message from the golden set or sample corpus retrieves its own historical resolution, creating an artificial shortcut for reply generation.
- **Trade-off / downside**: Slightly increases retrieval index bookkeeping by requiring query-time identity checks and mask manipulation across similarity matrices.
- **Evidence in current implementation**: [`src/retrieval/similar_threads.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/retrieval/similar_threads.py#L46-L53), which checks `if candidate_id == exclude_thread_id: continue` before populating top-$k$ results.

---

### 6. Retrieval-Constrained Generation with Empathy Prompts
- **Decision**: Conditioned reply generation strictly on the top-3 retrieved historical resolutions, explicitly prompting the model to state in-app next steps and forbidding the invention of specific dollar amounts or promises.
- **Why we chose it**: Grounding generative replies in verified corporate precedent eliminates catastrophic hallucinations (such as promising full refunds or quoting incorrect fare policies) while maintaining empathetic language.
- **Trade-off / downside**: When retrieved threads are unhelpful or tangentially related, the drafted reply can become repetitive, generic, or over-cautious, directing users to generic help links.
- **Evidence in current implementation**: [`src/generation/reply_generator.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/generation/reply_generator.py#L22-L38), where the prompt explicitly instructs: "Do not invent policies or make promises not supported by the evidence."

---

### 7. Decoupled Programmatic Escalation Engine vs. Generative LLM Discretion
- **Decision**: Isolated escalation decisions into a separate, deterministic Python module (`decide_escalation`) rather than allowing the LLM to decide whether to escalate in its completion prompt.
- **Why we chose it**: Safety boundaries and escalation compliance must be 100% auditable, deterministic, and immune to prompt injection or model mood/temperature variance.
- **Trade-off / downside**: Static rule sets lack subtle contextual reasoning, resulting in misclassifying nuanced frustration or novel grievance patterns that do not match hardcoded triggers.
- **Evidence in current implementation**: [`src/escalation/decide.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/escalation/decide.py#L12-L42), which applies sequential if/else checks on safety keywords, intents, and confidence thresholds.

---

### 8. Hardcoded Keyword Safety Triggers for Immediate Human Handover
- **Decision**: Established a hardcoded keyword list (`police`, `assault`, `emergency`, `accident`, `legal`, `lawyer`, `crash`, `injury`) that automatically triggers escalation regardless of classifier intent or confidence.
- **Why we chose it**: Guarantees zero-tolerance protection against automated handling of high-liability scenarios, ensuring legal and personal safety complaints are immediately routed to human specialists.
- **Trade-off / downside**: Causes over-escalation on benign or metaphorical usage (e.g., customer saying "my app crashed" or "your driver almost had an accident").
- **Evidence in current implementation**: [`src/escalation/decide.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/escalation/decide.py#L15-L23), which scans `text.lower()` for explicit safety tokens and forces `action="escalate"`.

---

### 9. Defensive Escalation via Strict Confidence Thresholding ($< 0.60$)
- **Decision**: Enforced an escalation trigger whenever intent classifier confidence drops below `0.60`.
- **Why we chose it**: Operates on a "fail-safe" principle: when the automated pipeline is uncertain about a customer's underlying problem, it should hand off to a human rather than issue a potentially irrelevant automated response.
- **Trade-off / downside**: Resulted in 46 over-escalation errors on short, polite, or low-context messages (e.g., "Thanks", "done") where low classifier confidence forced human intervention unnecessarily.
- **Evidence in current implementation**: [`src/escalation/decide.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/escalation/decide.py#L29-L33), which marks `action="escalate"` with reason `"Escalated because classifier confidence ... is below 0.60"`.

---

### 10. Configurable Regex Baseline with Fixed Auditable Templates
- **Decision**: Implemented `baselines/simple.py` using 50 data-derived keywords from `baselines/intent_keywords.json` and paired them with static, pre-approved reply templates.
- **Why we chose it**: Provided a realistic benchmark of what a zero-latency, zero-cost deterministic rule engine achieves, verifying whether an LLM actually adds net business value over classic keyword matching.
- **Trade-off / downside**: Regex patterns require manual maintenance and brittle curation, scoring low on intent recall when customers use unexpected synonyms or colloquialisms.
- **Evidence in current implementation**: [`baselines/simple.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/baselines/simple.py#L14-L46) and [`baselines/intent_keywords.json`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/baselines/intent_keywords.json).

---

### 11. Majority-Class Trivial Baseline for Skew Calibration
- **Decision**: Created `baselines/trivial.py`, which constantly predicts the empirical training majority intent (`Payment & Charges`) and always escalates (`escalate`).
- **Why we chose it**: Established the true floor for multi-class classification and demonstrated that high raw accuracy in unbalanced datasets can be deceptive without Macro F1 tracking.
- **Trade-off / downside**: The trivial baseline serves purely as an evaluation sanity check and has zero utility in production customer support.
- **Evidence in current implementation**: [`baselines/trivial.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/baselines/trivial.py#L13-L22), which calculates the majority class dynamically and outputs a constant prediction vector.

---

### 12. Explicit Blank-Label Evaluation Policy for Ambiguous Data
- **Decision**: Permitted 70 out of 200 golden-set rows to remain blank for ground-truth intent, computing intent accuracy and Macro F1 strictly over the 130 non-empty annotated rows.
- **Why we chose it**: Acknowledged that ambiguous, fragmented multi-turn utterances (e.g., "I did !", "nothing new") have no single objective intent, preventing artificial ground truth from distorting model evaluation.
- **Trade-off / downside**: Causes the intent evaluation denominator ($N=130$) to differ from the escalation evaluation denominator ($N=200$), which can confuse superficial readers if not explicitly explained.
- **Evidence in current implementation**: [`evaluation/harness.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/evaluation/harness.py#L56-L64), which conditionally appends pairs to `intent_pairs` only when `true_intent` is non-empty.

---

### 13. Deterministic Seed Pinning and Provider-Agnostic LLM Client Abstraction
- **Decision**: Pinned all random sampling across the project (`seed=42` for dataset sampling, `seed=7` for human agreement template selection) and abstracted LLM calls behind a generic OpenAI-compatible interface.
- **Why we chose it**: Pinned seeds ensure bit-for-bit test repeatability across environments, while the flexible endpoint wrapper enabled instant fallback to a local Ollama daemon (`llama3.2:latest`) when commercial API quotas or network limits were reached.
- **Trade-off / downside**: Local models run at lower tokens-per-second throughput on standard developer hardware compared to hosted cloud API clusters.
- **Evidence in current implementation**: [`src/data_prep/subsample.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/data_prep/subsample.py#L49), [`evaluation/judge_agreement.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/evaluation/judge_agreement.py#L114), and [`src/common/llm.py`](file:///c:/Users/yadua/OneDrive/Documents/Ai%20Ass/AI%20Agent%20For%20Cs/src/common/llm.py#L35-L68).
