About:
XGBoost + SHAP pipeline predicting store-level customer satisfaction at CellphoneS using delivery performance, mystery shopper audits, and planogram compliance (n=427, R²=0.845).

# Predictive Modeling of Customer Satisfaction at CellphoneS

**Author:** Đoàn Trần Bửu Phúc (2212255037) - K61CLC1, International Business Administration  
**Supervisor:** PhD. Lê Trung Thành  
**Institution:** Foreign Trade University - Ho Chi Minh City Branch  
**Submitted:** June 2026 | Thesis ID: B40

---

## Overview

This thesis builds and evaluates machine learning models to predict **store-level customer satisfaction** at CellphoneS (~160 stores, Vietnam) using internal operational audit data - no customer surveys required.

**Unit of analysis:** Store × month panel  
**Sample:** 427 observations · 152 stores · 6 months (Oct 2025–Mar 2026)  
**Dependent variable:** Weighted satisfaction score `(5·satisfied + 3·neutral + 1·dissatisfied) / total_tickets`

---

## Independent Variables

| Pillar | Framework | Key features |
|---|---|---|
| Mystery Shopper (IV1) | RSQS - Dabholkar et al. (1996) | PA, R, PI, PS, PO sub-dimension pass rates |
| Delivery Performance (IV2) | LSQ - Mentzer et al. (2001) | `delay_max_hours`, `ontime_rate`, `promise_gap_mean_hours` |
| Planogram Compliance (IV3) | Service-Profit Chain - Heskett et al. (1994) | `planogram_score`, error sub-categories |

---

## Models & Results

| Model | Test R² | CV R² (mean ± SD) | Test MAE |
|---|---|---|---|
| Baseline (mean) | −0.007 | −0.031 ± 0.042 | 1.150 |
| Decision Tree | 0.372 | 0.549 ± 0.105 | 0.471 |
| Random Forest | 0.692 | 0.761 ± 0.089 | 0.437 |
| **XGBoost** | **0.845** | **0.769 ± 0.092** | **0.334** |

Tuning: `RandomizedSearchCV` · Validation: 80/20 split + 5-fold CV · Explainability: SHAP TreeExplainer

---

## Key Finding

`delay_max_hours` (worst-case delivery delay) is the single dominant predictor - Mean |SHAP| = 0.729, consistent across all three tree models. Delivery features collectively explain **60–71% of total SHAP importance**. Mystery Shopper features contribute 4–9%, with only the Problem Solving sub-dimension showing meaningful signal.

---

## Stack

`Python` · `XGBoost` · `scikit-learn` · `SHAP` · `pandas` · `matplotlib`
