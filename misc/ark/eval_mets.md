# No-LLM Evaluation Metrics for SecArch Pipeline

## Evaluation Metric Taxonomy

### Correctness
> *"Is the output factually accurate against a known ground truth?"*

- Technique ID exists in MITRE ATT&CK
- Technique name matches canonical MITRE name
- Mitigation ID exists in MITRE
- Mitigation name matches canonical name
- **Mitigation is actually linked to that technique in MITRE** ← strongest signal

This is the most rigorous dimension here because MITRE is an authoritative, verifiable ground truth.

---

### Groundedness
> *"Is the output anchored to the provided inputs, or is it hallucinating?"*

- Component recall: mentioned components appear in output
- Perimeter word recall: words from input appear in `perimeter_words_identified`
- No new technique IDs invented in Stage 2/3 that weren't in Stage 1
- Internet-facing / data sensitivity context reflected in attack selection

Groundedness ≠ correctness — an output can be factually correct but disconnected from *this specific architecture*.

---

### Completeness
> *"Does the output cover everything it should?"*

- Every Stage 1 technique appears in Stage 2 attack trees
- Every Stage 1 technique has mitigations in Stage 3
- Every mitigation has a reasoning entry
- Minimum attack count given a high-risk input profile
- Every mitigation maps to ≥ 1 internal control

---

### Structural Validity *(Format Adherence)*
> *"Does the output conform to the expected schema?"*

- Pydantic model parse success
- Numbered steps regex check
- `len(mitigation_ids) == len(reasonings)`
- No duplicate technique IDs

This is usually treated separately from the semantic dimensions — it is a precondition for the others being computable at all.

---

### Consistency
> *"Do outputs agree with each other across the pipeline stages?"*

- Technique IDs are the same set across all 3 stages
- Attack classification does not contradict tactic type (e.g. Phishing ≠ Internal)

This is a **cross-stage** dimension unique to multi-turn/chained pipelines like this one.

---

## Additional Metrics

| Metric | What it measures | How |
|---|---|---|
| **Specificity** | Are descriptions generic boilerplate or architecture-specific? | TF-IDF or token overlap between the description field and the input component names — low overlap = generic hallucination |
| **Calibration** | Does attack severity/frequency match the risk profile inputs? | Higher-risk inputs (HIGHLY PROTECTED + internet-facing) should produce more attacks and higher-severity techniques than PUBLIC + internal |
| **Recall vs. Expert Baseline** | How many of the attacks a human expert would flag did the model find? | Requires a manually labelled "golden set" of a few architecture diagrams — high effort but gives an absolute quality floor |
| **False Positive Rate** | Are attacks flagged that are clearly inapplicable given the architecture? | Partially checkable: e.g. if no cloud components are present, flagging T1537 *Transfer Data to Cloud Account* is a likely false positive |
| **Control Coverage Gap** | What % of mitigations have no mapped internal control? | Pure data: `empty control_ids / total mitigation rows` from the CSV — tells you where your controls dictionary has holes |
| **Latency / Token Efficiency** | Cost vs. quality tradeoff across model calls | Already partially in `LLMLog` — correlate token count with eval scores |

---

## Metric Hierarchy

```
Structural Validity  ← must pass first (prerequisite)
       ↓
Correctness          ← highest trust signal (MITRE ground truth)
       ↓
Groundedness         ← architecture-specificity signal
       ↓
Completeness         ← coverage signal
       ↓
Consistency          ← pipeline integrity signal
       ↓
Specificity          ← quality-of-reasoning signal (hardest to automate)
```

> **Note on Coherence:** Typically measures natural language fluency — less meaningful here since outputs are structured JSON with short reasoning strings, not prose. Not a priority metric for this pipeline.

---

## Checks by Pipeline Stage

### Stage 1 — `identify_architecture_gaps`

| Check | Metric Dimension |
|---|---|
| Schema parses via `IdentifiedOutputFormat` | Structural Validity |
| Every `technique_id` exists in MITRE STIX bundle | Correctness |
| `technique_name` matches canonical name for that ID | Correctness |
| `perimeter_words_identified` ∩ input perimeter words | Groundedness / Completeness |
| Input component tokens appear in output component lists | Groundedness |
| ≥ N attacks for high-risk input profile | Completeness / Calibration |
| No duplicate technique IDs across attack classes | Structural Validity |

### Stage 2 — `generate_attack_trees`

| Check | Metric Dimension |
|---|---|
| Every Stage 1 `technique_id` is present | Completeness / Consistency |
| No new technique IDs introduced | Groundedness |
| All attacks classified into a category | Completeness |
| `steps` is a non-empty list of non-empty strings | Structural Validity |
| Each step matches regex `^\d+\.` | Structural Validity |
| Tactic type consistent with classification bucket | Consistency |

### Stage 3 — `generate_mitigations`

| Check | Metric Dimension |
|---|---|
| Every `Mxxxx` exists in MITRE STIX bundle | Correctness |
| Mitigation name matches canonical name | Correctness |
| M-ID is linked to that technique in MITRE `mitigates` relationships | Correctness |
| Every Stage 1 technique has an entry | Completeness / Consistency |
| `len(reasonings) == len(mitigation_ids)` | Structural Validity |
| No empty reasoning strings | Completeness |

### Stage 4 — IAG Controls Mapping

| Check | Metric Dimension |
|---|---|
| All `control_ids` exist in `rev_mapping_manual_mit_to_controls_final.json` | Correctness |
| Every mitigation row has ≥ 1 mapped control | Completeness |
| M-IDs with no controls flagged as dictionary gaps | Control Coverage Gap |
