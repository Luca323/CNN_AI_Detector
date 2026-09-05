# Post-Implementation Audit

> Comparing the current codebase against [analysis_results.md](file:///c:/Users/Luca-/Documents/my-app/analysis_results.md)

---

## Scorecard — Original Findings

| # | Finding | Verdict | Notes |
|---|---------|---------|-------|
| §1.1 | Validation overuse leak (3 roles for 1 split) | ✅ **FIXED** | [train.py L82-88](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L82-L88): Now splits into 4 partitions (70/10/10/10). Val drives model selection + LR scheduling only. Calibration split handles threshold tuning at [L165-188](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L165-L188). Test is fully isolated at [L190-206](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L190-L206). Clean separation. |
| §1.2 | Root-TTR magic constant `3.5` | ❌ **NOT ADDRESSED** | [feature_extractor.py L45](file:///c:/Users/Luca-/Documents/my-app/python/feature_extractor.py#L45): Still uses undocumented `/ 3.5`. No provenance comment. |
| §1.3 | Feature extraction before splitting | ✅ **FIXED** | [train.py L82-96](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L82-L96): Raw DataFrames are split first, then each partition is extracted independently via `extract_dataset_features()`. |
| §2.1 | `torch.load()` missing `weights_only` | ⚠️ **PARTIALLY FIXED** | [train.py L167](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L167): ✅ Fixed. [api.py L87](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L87): ✅ Fixed. Both now use `weights_only=True`. |
| §2.2 | Stale hardcoded fallback metrics in API | ✅ **FIXED** | [api.py L56-74](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L56-L74): Startup now raises `FileNotFoundError` if checkpoint or metrics.json is missing — zero fallback policy. [api.py L108-118](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L108-L118): `/model-info` also enforces file existence. |
| §2.3 | API uses fixed `0.50` threshold | ✅ **FIXED** | [api.py L37](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L37): `optimal_threshold` global initialised as 0.50. [api.py L92-95](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L92-L95): Overwritten from `metrics.json` at startup. [api.py L200](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L200): Classification uses `optimal_threshold`. |
| §2.4 | Architecture docs outdated | ✅ **FIXED** | [api.py L122-138](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L122-L138): Now includes `k=1` pointwise conv, correct `Dense(1284 -> 512 -> 128 -> 1) with LayerNorm`. |
| §3.1 | Sentence scores are fake heuristics | ✅ **FIXED** | [api.py L142-179](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L142-L179): Sinusoidal/heuristic code fully removed. Each sentence now gets its own independent `feature_extractor.extract_features()` + `detector_model()` forward pass. |
| §3.2 | `get_sliding_window_scores()` never used | ⚠️ **PARTIALLY FIXED** | [model.py L93-130](file:///c:/Users/Luca-/Documents/my-app/python/model.py#L93-L130): Method improved with mask support, valid-length trimming, and empty-token guard. Still **not wired** into the API — the API runs full-model inference per sentence instead. Approach is valid but different from the recommendation. |
| §3.3 | Perplexity proxy is redundant | ❌ **NOT ADDRESSED** | [feature_extractor.py L68-69](file:///c:/Users/Luca-/Documents/my-app/python/feature_extractor.py#L68-L69): Still `1.0 - (repetition_index * 1.2) + (sentence_variation * 0.2)`. Still a deterministic linear combo of other features. |
| §3.4 | Confidence bands ignore threshold | ❌ **NOT ADDRESSED** | [api.py L202-207](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L202-L207): Still symmetric around 0.50 (`>= 0.85`, `>= 0.70`). Not calibrated to the loaded `optimal_threshold`. |
| §4.1 | No gradient clipping | ✅ **FIXED** | [train.py L131](file:///c:/Users/Luca-/Documents/my-app/python/train.py#L131): `clip_grad_norm_(model.parameters(), max_norm=1.0)` added. |
| §4.2 | `max_length=128` too short | ❌ **NOT ADDRESSED** | Still 128 everywhere. |
| §4.3 | No reproducibility seeds | ❌ **NOT ADDRESSED** | No `torch.manual_seed()` or `np.random.seed()` calls before training. |

### Summary

| Status | Count |
|--------|-------|
| ✅ Fixed | 8 / 14 |
| ⚠️ Partially Fixed | 2 / 14 |
| ❌ Not Addressed | 4 / 14 |

All three **CRITICAL** items (§1.1, §2.3, §3.1) are fixed. Good.

---

## New Bugs & Issues Introduced

### NEW-1 🐛 Silent Exception Swallowing in Sentence Scoring
**Severity: MEDIUM — Silent Data Corruption**

| File | Lines |
|------|-------|
| [api.py](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L159-L161) | L159-161 |

```python
except Exception:
    # If sentence extraction fails (e.g. punctuation only), evaluate sentence text length
    s_ai_prob = 0.50
```

The bare `except Exception` catches **all** failures — CUDA OOM, model weight corruption, tokenizer errors — and silently returns `0.50` (classified as "Medium" risk). This means:

- A catastrophic failure on 1 sentence doesn't surface to the user or logs
- The fallback `0.50` is presented as a real model prediction
- The comment says "evaluate sentence text length" but it doesn't — it just hardcodes 0.50

Meanwhile, [feature_extractor.py L39](file:///c:/Users/Luca-/Documents/my-app/python/feature_extractor.py#L39) now raises `ValueError` on empty text, which means punctuation-only sentences **will** hit this catch block routinely, mixing genuine errors with expected edge cases.

> [!WARNING]
> **Fix:** Catch only `ValueError` for known edge cases (empty/punctuation-only sentences). Let CUDA and model errors propagate. Log caught exceptions. Consider returning a distinct flag (`"risk": "Unavailable"`) instead of a fake score.

---

### NEW-2 ⚠️ Sentence-Level Risk Bands Not Calibrated to Threshold Either
**Severity: LOW — Inconsistency with §3.4**

| File | Lines |
|------|-------|
| [api.py](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L163-L170) | L163-170 |

The sentence-level risk classification uses hardcoded bands:
```python
if s_ai_prob >= 0.75:   risk = "Critical"
elif s_ai_prob >= 0.55: risk = "High"
elif s_ai_prob >= 0.35: risk = "Medium"
else:                   risk = "Low"
```

These are the same uncalibrated bands as the old code. With `optimal_threshold=0.35`, a sentence at `s_ai_prob=0.40` is labelled "Medium" risk even though the model considers it a positive classification. The sentence-level and document-level risk signals are inconsistent.

---

### NEW-3 ⚠️ `feature_extractor.py` Raises on Empty Text — But Single-Sentence Default Bias Changed
**Severity: LOW — Behavioral Change**

| File | Lines |
|------|-------|
| [feature_extractor.py](file:///c:/Users/Luca-/Documents/my-app/python/feature_extractor.py#L38-L39) | L38-39 |
| [feature_extractor.py](file:///c:/Users/Luca-/Documents/my-app/python/feature_extractor.py#L66) | L66 |

Two changes:
1. **L38-39**: Empty text now raises `ValueError` instead of returning `[0.5, 0.0, 0.2, 0.5]`. This is a good defensive change, but nothing in the training pipeline guards against it — if a sample in the RAID dataset contains whitespace-only text, training will crash.

2. **L66**: Single-sentence `sentence_variation` default changed from `0.35` to `0.0`. This is a more honest default (no variation = zero), but it changes the feature distribution compared to any previously trained model. **If the current checkpoint was trained with `0.35`, this creates a train/serve skew.**

> [!IMPORTANT]
> If you retrain after this change, the `0.0` default is correct. If you deploy with the **existing** checkpoint (trained on `0.35`), this is a silent distribution shift at inference time.

---

### NEW-4 ⚠️ Performance Regression: N Forward Passes per Prediction
**Severity: MEDIUM — Latency**

| File | Lines |
|------|-------|
| [api.py](file:///c:/Users/Luca-/Documents/my-app/python/api.py#L153-L158) | L153-158 |

The new `analyze_sentences()` runs a **separate** BERT encoding + CNN forward pass per sentence. For a document with 20 sentences, this means:
- 1 document-level inference (existing)
- 20 sentence-level inferences (new)
- **21 total forward passes** vs 1 previously

BERT tokenization + forward pass is ~15-50ms per call on GPU. A 20-sentence document now takes ~300-1000ms in `analyze_sentences()` alone. This is functionally correct but a significant latency regression that should be batched.

> [!TIP]
> **Fix:** Batch all sentences into a single `extract_features(raw_sentences, max_length=128)` call + single `detector_model()` forward pass.

---

## Revised Priority Matrix

```
STILL OPEN — Should Fix
├── §3.3  Perplexity proxy redundancy (multicollinearity still present)
├── §3.4  Confidence bands still hardcoded around 0.50
├── NEW-1  Silent exception swallowing masks real errors
├── NEW-3  sentence_variation default 0.35→0.0 creates train/serve skew
└── NEW-4  N forward passes per prediction (batch the sentences)

NICE TO HAVE
├── §1.2  Document the Root-TTR 3.5 divisor provenance
├── §4.2  Extend max_length to 256
├── §4.3  Add reproducibility seeds
└── NEW-2  Sentence risk bands not calibrated to threshold
```

> [!NOTE]
> Overall the changes are strong. The three critical issues from the original audit are all properly resolved, and the split-before-extract refactor in `train.py` is clean. The main new concern is the silent exception catch in sentence scoring and the latency cost of per-sentence inference.
