# 🏦 Credit Risk Scoring with a Deep Hybrid Ensemble

**A Random Forest + XGBoost + Neural Network stacking ensemble for predicting loan default, built to survive scrutiny, not just report a number.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-NN-red)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-ff4b4b)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## Why this project exists

Most public notebooks on this dataset stop at "trained a model, got 0.86 AUC." That number alone tells a lender almost nothing about whether the model is *usable*. This project asks the harder questions a real deployment would force you to answer:

- Does a fancier ensemble actually beat a well-tuned single model, or just look good on paper?
- What decision threshold should the business actually use, and why isn't 0.5 the right answer?
- Can you trust the model's probability outputs, or only its rankings?
- Can you explain *why* any single applicant was flagged, not just that they were?

Every one of those questions gets a real, evidence-backed answer below.

---

## The short version

| | |
|---|---|
| **Dataset** | [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 applicants, 93.3% / 6.7% class imbalance |
| **Best single model** | Neural Network — 0.8446 AUC-ROC |
| **Hybrid ensemble** | **0.8497 AUC-ROC** — beats every baseline |
| **Cost-optimal threshold** | 0.33 (vs. naive 0.5) → catches **55%** of defaulters instead of 33% |
| **Explainability** | SHAP, global + per-applicant |
| **Deployment** | Live Streamlit app with adjustable risk threshold |

---

## The Problem

Lenders need to predict which applicants will become seriously delinquent within two years, and act on that prediction in a way that minimizes real financial cost, not just maximizes accuracy. This project builds and rigorously evaluates a credit risk model with a focus on the parts most tutorials skip: honest baselines, leakage-free ensembling, cost-sensitive decision thresholds, and explainability.

## Dataset

- **Source**: Kaggle, "Give Me Some Credit"
- **Size**: 150,000 rows, 10 raw features
- **Target**: `SeriousDlqin2yrs` — whether the applicant was 90+ days delinquent within 2 years
- **Class balance**: 93.32% no default / 6.68% default (heavily imbalanced)

---

## Approach

### 1. Data Cleaning
- Removed a single invalid row (`age = 0`)
- Capped extreme outliers in utilization ratio, debt ratio, and past-due counts, rather than deleting affected rows, to preserve genuine high-risk signal
- Identified 269 rows containing sentinel/placeholder values (96/98) in past-due columns; preserved this as a separate engineered flag (`had_sentinel_value`) instead of letting it silently distort the model
- Imputed missing `MonthlyIncome` (19.8% missing) with median, `NumberOfDependents` (2.6% missing) with mode

### 2. Baseline Models
Four models trained and tuned independently, with SMOTE applied only within cross-validation folds to prevent leakage:

| Model | AUC-ROC |
|---|---|
| **Neural Network (PyTorch)** | **0.8446** |
| Random Forest (tuned) | 0.8379 |
| Logistic Regression | 0.8374 |
| XGBoost (tuned) | 0.8341 |

All four land within a tight 0.011 band, none has an unfair advantage, which makes the ensemble result below a genuine test of whether stacking adds value.

### 3. Hybrid Ensemble (Stacking)

```
Random Forest ─┐
XGBoost ────────┼──► Out-of-Fold Predictions ──► Logistic Regression Meta-Learner ──► Final Score
Neural Network ─┘
```

- Out-of-fold (OOF) predictions generated via 5-fold cross-validation, so the meta-learner never sees predictions on data a base model was trained on
- Final base models retrained on the full training set to score the untouched test set

> **Result: AUC-ROC 0.8497** — beats every individual baseline.
> Meta-learner weights (RF: 1.84, XGBoost: 1.60, NN: 2.33) show all three base models contribute real, non-redundant signal — no single model is carrying the ensemble.

### 4. Business-Aware Evaluation

A 0.5 threshold is the default almost everyone uses, and it's almost never the right one. Assuming a missed defaulter (false negative) costs 5x a wrongly rejected applicant (false positive):

| Threshold | Defaulters Caught | False Positives | Accuracy |
|---|---|---|---|
| 0.50 (naive default) | 33% | 854 | 93% |
| **0.33 (cost-optimal)** | **55%** | 2,440 | 89% |

At 0.33, the model catches **443 more real defaulters** — a direct, quantifiable win — at the cost of more false alarms. Accuracy actually *drops* (93% → 89%), which is exactly the point: accuracy is the wrong metric to chase in an imbalanced, cost-asymmetric problem like this one.

A calibration check (Brier score 0.0583) showed the model is systematically overconfident, most likely from SMOTE shifting the base models' learned class balance. Isotonic regression gave only a marginal fix (0.0577). Documented honestly as a limitation: the model's **rankings** are trustworthy, its raw probabilities are not, yet, without further per-base-model recalibration.

### 5. Explainability (SHAP)

SHAP applied to the Random Forest base model (XGBoost skipped — version incompatibility between XGBoost 3.x and the SHAP `TreeExplainer` parser, documented rather than hidden).

- **Top global driver**: `RevolvingUtilizationOfUnsecuredLines` — matches domain intuition
- **Per-applicant waterfall plots** show precisely which factors drove any single risk score, supporting the kind of adverse-action explanation lenders are legally required to give rejected applicants

### 6. Interactive Deployment (Streamlit)

A working app where a user enters an applicant's details and gets, live, from the full hybrid pipeline:
- Predicted default probability
- An approve / flag decision at the cost-optimal 0.33 threshold
- An **adjustable threshold slider**, showing the decision isn't fixed, it's a business policy lever
- A breakdown of what each individual base model predicted, and how the meta-learner combined them

**Tested end-to-end:**
| Profile | Predicted Risk | Decision |
|---|---|---|
| Low-risk applicant | 3.7% | ✅ Approved |
| High-risk applicant (real defaulter from test set) | 79.0% | ⚠️ Flagged |

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · PyTorch · imbalanced-learn (SMOTE) · SHAP · Streamlit

## Project Structure

```
credit-risk-hybrid-ensemble/
├── cs-training.csv                     # raw dataset
├── credit_risk_hybrid_ensemble.ipynb   # full analysis, modeling, evaluation
├── streamlit_app.py                    # interactive risk scoring app
├── rf_final_model.pkl                  # trained Random Forest
├── xgb_final_model.pkl                 # trained XGBoost
├── nn_final_model.pth                  # trained PyTorch neural network weights
├── meta_model.pkl                      # trained stacking meta-learner
├── scaler_nn.pkl                       # fitted scaler for the neural network
└── README.md
```

## Running the App Locally

```bash
conda activate creditrisk
cd credit-risk-hybrid-ensemble
streamlit run streamlit_app.py
```

---

## Key Takeaways

- A properly validated hybrid ensemble gives a modest but genuine lift over the best single model (+0.005 AUC-ROC) — a realistic result, not an inflated one
- Threshold selection should be driven by business cost, not left at a default 0.5 cutoff
- Ranking performance and probability calibration are two different things; a model can be great at one and weak at the other
- Explainability isn't optional in regulated domains like lending — SHAP gives both a global and an individual-level answer to "why"

## Limitations

- Dataset is a static historical snapshot with no time-series/trend information per applicant
- Probability calibration was only partially correctable; a production system would need per-base-model recalibration before stacking
- The 5:1 cost ratio used for threshold optimization is illustrative; a real deployment would use actual institutional loss data
