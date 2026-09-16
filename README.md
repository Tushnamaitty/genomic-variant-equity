# Explainable ML for Longitudinal VUS Resolution and Population-Aware Model Auditing

An AI/ML pipeline for predicting whether a **ClinVar Variant of Uncertain Significance (VUS)** will later receive a resolved clinical interpretation, with explainability, probability calibration, and population-aware model auditing.

This repository contains a **chromosome 22 proof-of-concept** integrating longitudinal ClinVar records with gnomAD v4.1 population-frequency data.

---

## Overview

Variants of Uncertain Significance are genetic variants for which currently available evidence is insufficient to determine whether they are benign or pathogenic.

As new clinical and scientific evidence becomes available, some VUS are later reclassified.

This project asks:

> **Can information available at an earlier ClinVar snapshot predict whether a VUS will later receive a resolved clinical interpretation?**

A secondary analysis investigates whether model explanations differ for variants enriched in **South Asian (SAS)** versus **non-Finnish European (NFE)** gnomAD populations.

The project combines:

- longitudinal genomic prediction
- real ClinVar and gnomAD data
- severe class-imbalance handling
- Logistic Regression and XGBoost
- cross-validated hyperparameter tuning
- SHAP explainability
- population-aware model auditing
- population-blind sensitivity analysis
- probability calibration

---

## Research Question

The primary ML task is:

> Given information available for a VUS at time T1, can we predict whether it will later be resolved at time T2?

The secondary model-auditing question is:

> Do the model's feature-attribution patterns differ between SAS-enriched and NFE-enriched variants?

This project does **not** attempt to predict whether a variant is pathogenic or benign.

Instead, it predicts whether a currently uncertain variant is likely to receive a resolved ClinVar interpretation in the future.

---

## Study Design

Two ClinVar snapshots were used:

- **T1:** February 2024
- **T2:** July 2025

Variants classified as **Uncertain significance** at T1 were followed longitudinally to T2.

The prediction target was defined as:

```text
T1: Variant of Uncertain Significance
                  |
                  v
          Follow variant to T2
                  |
          +-------+-------+
          |               |
          v               v
      Resolved         Still VUS
   benign/pathogenic
```

Variants with conflicting or other non-primary classifications at T2 were excluded from the primary ML endpoint.

---

## Dataset

### Chromosome 22 Cohort

| Stage | Number of Variants |
|---|---:|
| All chromosome 22 ClinVar records | 26,575 |
| GRCh38 records | 26,008 |
| GRCh38 variants with usable coordinates/alleles | 25,846 |
| Final clean longitudinal ML cohort | **25,294** |

### Target Distribution

| T2 Outcome | Count |
|---|---:|
| Still VUS | 25,008 |
| Resolved | **286** |

The positive class therefore represents only approximately **1.13%** of the final cohort.

This makes the problem a highly imbalanced binary-classification task.

---

## ClinVar Longitudinal Label Construction

At T1, all included variants were classified as:

```text
Uncertain significance
```

At T2, outcomes were categorized as:

- `resolved_benign`
- `resolved_pathogenic`
- `still_vus`
- `other`
- unmatched

The clean primary ML endpoint retained:

```text
resolved_benign
resolved_pathogenic
still_vus
```

The binary target was:

```text
is_resolved = 1
    if resolved_benign or resolved_pathogenic

is_resolved = 0
    if still_vus
```

Conflicting classifications and other non-primary outcomes were excluded.

---

## gnomAD Integration

ClinVar variants were normalized and matched against the **gnomAD v4.1 joint dataset**.

Population-specific information included:

- SAS allele frequency
- NFE allele frequency
- SAS allele count
- NFE allele count
- SAS allele number
- NFE allele number
- SAS-NFE allele-frequency difference
- Fisher exact-test odds ratio
- Fisher exact-test p-value

Variant matching used normalized:

```text
chromosome + position + REF + ALT
```

### Chromosome 22 Matching Results

| Result | Count |
|---|---:|
| Usable ClinVar variants | 25,846 |
| Exact gnomAD matches | 20,033 |
| Unmatched variants | 5,806 |
| Reference-N exclusions | 6 |
| REF = ALT exclusion | 1 |

The normalization and matching process was implemented as a reproducible chromosome-level pipeline.

---

## Population-Enrichment Analysis

Variants were compared across SAS and NFE gnomAD populations using allele counts and allele numbers.

For each testable variant, a two-sided Fisher exact test was applied using:

```text
             Alternate   Reference
SAS             AC       AN - AC
NFE             AC       AN - AC
```

Benjamini-Hochberg false-discovery-rate correction was then applied across chromosome 22.

Population-enrichment groups were defined as:

```text
SAS-enriched:
q < 0.05 AND odds ratio >= 2

NFE-enriched:
q < 0.05 AND odds ratio <= 0.5
```

Other variants were assigned to categories including:

- no clear enrichment
- observed zero in both populations
- unmatched
- insufficient data

### Clean ML Cohort

| Group | Variants | Resolved | Resolution Rate |
|---|---:|---:|---:|
| SAS-enriched | 1,709 | 16 | 0.94% |
| NFE-enriched | 1,408 | 9 | 0.64% |

These groups represent **population-frequency enrichment of variants**, not patient ancestry.

Because the number of resolved cases within these subgroups is small, subgroup outcome comparisons are treated as exploratory.

---

## Feature Engineering

Only information available at **T1** was allowed into the ML feature space.

All T2 information was excluded to prevent temporal leakage.

### ClinVar Features

- review status
- evaluation age in days
- missing evaluation-date indicator

### Variant-Structure Features

- SNV / insertion / deletion / other
- REF allele length
- ALT allele length
- length change
- nucleotide substitution
- multi-gene indicator

### gnomAD Population Features

- SAS allele frequency
- NFE allele frequency
- SAS allele count
- NFE allele count
- SAS allele number
- NFE allele number
- SAS-NFE AF difference
- absolute AF difference
- Fisher odds ratio
- Fisher p-value
- gnomAD match status
- population-data missingness indicators

---

## Train-Test Split

The final ML cohort was divided using a fixed **80/20 stratified train-test split**.

```text
Training set: 20,235 variants
    20,006 unresolved
       229 resolved

Test set: 5,059 variants
     5,002 unresolved
        57 resolved
```

The positive prevalence remained approximately **1.13%** in both sets.

The test set was kept untouched during hyperparameter tuning.

---

## Machine-Learning Models

Three main models were evaluated.

### 1. Logistic Regression

A class-weighted Logistic Regression model was used as the interpretable baseline.

### 2. Full XGBoost

The full model included:

- ClinVar features
- variant-structure features
- gnomAD population features

Class imbalance was handled using `scale_pos_weight`.

### 3. Population-Blind XGBoost

A sensitivity model was trained after removing all explicit gnomAD and population-derived features.

This model retained only:

```text
ClinVar history
+
variant characteristics
```

This allowed assessment of whether explicit population-frequency information was necessary for predictive performance.

---

## Evaluation Metrics

Because only approximately 1.13% of variants resolved, overall accuracy is not informative.

Primary evaluation focused on:

- **PR-AUC / Average Precision**
- ROC-AUC
- Precision
- Recall
- F1 score
- Brier score
- probability calibration

For this dataset, the positive-prevalence PR baseline is approximately:

```text
0.0113
```

---

## XGBoost Hyperparameter Tuning

XGBoost was tuned using:

- 5-fold stratified cross-validation
- randomized hyperparameter search
- average precision / PR-AUC as the optimization metric

Best mean cross-validation PR-AUC:

```text
0.0360
```

The selected parameters were:

```text
n_estimators      = 400
max_depth         = 2
learning_rate     = 0.05
min_child_weight  = 3
subsample         = 1.0
colsample_bytree  = 1.0
gamma             = 0
```

The shallow tree depth also improves interpretability and reduces overfitting risk.

---

## Model Performance

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6490 | 0.0171 |
| Full XGBoost | 0.7348 | 0.0296 |
| **Population-Blind XGBoost** | **0.7406** | **0.0346** |

The population-blind model achieved approximately **3× the positive-prevalence PR baseline**.

The tuned full XGBoost also substantially outperformed the Logistic Regression baseline.

---

## Logistic Regression Baseline

The clean Logistic Regression baseline achieved:

```text
ROC-AUC:   0.6490
PR-AUC:    0.0171
Precision: 0.0173
Recall:    0.6842
F1:        0.0338
```

This provided a reference point for evaluating nonlinear gradient-boosted models.

---

## Tuned Full XGBoost

The tuned full model achieved:

```text
ROC-AUC:   0.7348
PR-AUC:    0.0296
Precision: 0.0245
Recall:    0.5965
F1:        0.0472
```

The corresponding test confusion matrix at a threshold of 0.5 was:

```text
TN = 3651
FP = 1351
FN =   23
TP =   34
```

PR-AUC improved substantially compared with Logistic Regression.

---

## Threshold Analysis

A classification threshold was selected using out-of-fold predictions from the training set only.

The best F1 threshold on training CV predictions was:

```text
0.78
```

Training-CV metrics at this threshold were:

```text
Precision: 0.0601
Recall:    0.0830
F1:        0.0697
```

When the frozen threshold was applied to the held-out test set:

```text
Precision: 0.0606
Recall:    0.0702
F1:        0.0650
```

Because threshold-dependent metrics are unstable with only 57 positive test examples, the project emphasizes threshold-independent metrics such as PR-AUC and ROC-AUC.

---

## Population-Blind Sensitivity Model

The population-blind model excluded:

- gnomAD match status
- SAS AF
- NFE AF
- SAS AC
- NFE AC
- SAS AN
- NFE AN
- odds ratio
- Fisher p-value
- SAS-NFE AF difference
- absolute AF difference
- population comparison category
- population missingness indicators

Its test performance was:

```text
ROC-AUC: 0.7406
PR-AUC:  0.0346
```

This slightly exceeded the full model:

```text
Full XGBoost PR-AUC:             0.0296
Population-blind XGBoost PR-AUC: 0.0346
```

Within this chromosome 22 pilot, explicit population-frequency features were therefore **not necessary for predictive performance**.

Because the held-out test set contains only 57 resolved variants, this difference should be interpreted cautiously and not as evidence that population-frequency information universally reduces performance.

---

## Explainable AI with SHAP

SHAP was used to explain the tuned XGBoost model.

Mean absolute SHAP values quantified how strongly each feature influenced model predictions.

### Most Important Full-Model Features

The strongest features included:

1. `evaluation_age_days`
2. `AN_joint_sas`
3. `AN_joint_nfe`
4. ClinVar review status
5. `af_difference_sas_minus_nfe`
6. `AF_joint_sas`
7. variant type
8. `AF_joint_nfe`
9. absolute AF difference
10. `AC_joint_nfe`

The strongest feature overall was:

```text
evaluation_age_days
```

The full model therefore relied strongly on both:

- ClinVar evidence history
- population-data availability

Population allele-number variables were particularly important, suggesting the model was sensitive not only to observed allele frequencies but also to the amount of population data available.

---

## SHAP Interpretation

The SHAP beeswarm analysis indicated that **evaluation age** was the dominant feature influencing future-resolution predictions.

Population-data coverage variables such as:

```text
AN_joint_sas
AN_joint_nfe
```

also made substantial contributions.

This is important because allele number can reflect dataset coverage and evidence availability in addition to biological variation.

Therefore, these features should not automatically be interpreted as biological effects.

---

## Population-Aware SHAP Audit

SHAP explanations were compared between:

- SAS-enriched variants
- NFE-enriched variants

Held-out subgroup sizes were:

```text
SAS-enriched: 345
NFE-enriched: 285
```

In the full model, some of the largest SHAP differences occurred in:

- SAS-NFE AF difference
- NFE AF
- SAS AF
- review status
- SAS allele number
- odds ratio
- evaluation age

However, SAS/NFE enrichment itself was defined using population-frequency statistics.

Therefore, large SHAP differences involving those same population-derived features are partly expected by construction and were **not interpreted as evidence of model bias**.

---

## Population-Blind SHAP Audit

To produce a cleaner comparison, subgroup SHAP analysis was repeated using the population-blind model.

Because this model could not directly access population-frequency features, any remaining differences arose from ClinVar-history and variant-structure features.

### Largest Remaining Differences

#### ClinVar Review Status

Mean absolute SHAP:

```text
SAS-enriched: 0.2552
NFE-enriched: 0.3528
```

#### Evaluation Age

```text
SAS-enriched: 0.9592
NFE-enriched: 1.0224
```

#### Multi-Gene Indicator

```text
SAS-enriched: 0.0505
NFE-enriched: 0.0767
```

Other variant-type and substitution differences were comparatively small.

The population-blind audit therefore showed that subgroup explanation differences became much smaller after explicit population features were removed.

The remaining differences were primarily associated with:

- ClinVar evidence history
- evaluation timing
- variant characteristics

These findings are exploratory and **do not establish algorithmic bias, biological disparity, or causal inequity**.

---

## Probability Calibration

The population-blind XGBoost produced poorly calibrated raw probabilities because class weighting strongly altered the model's probability scale.

### Uncalibrated Model

```text
ROC-AUC:     0.7406
PR-AUC:      0.0346
Brier score: 0.187140
```

The raw model was substantially overconfident.

For example, some probability bins had mean predicted probabilities above 0.5 despite observed resolution rates remaining only a few percent.

Two calibration approaches were therefore evaluated:

- Platt scaling
- isotonic regression

### Calibration Results

| Method | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Uncalibrated | 0.7406 | 0.0346 | 0.187140 |
| **Platt Scaling** | **0.7413** | 0.0333 | **0.011037** |
| Isotonic | 0.7377 | **0.0358** | 0.011093 |

Platt scaling was selected as the primary calibration method because it produced the lowest Brier score while preserving discrimination.

Calibration reduced the Brier score from:

```text
0.1871
   ↓
0.0110
```

This demonstrates that probability calibration is essential when using class-weighted models for extremely imbalanced genomic outcomes.

---

## Key Findings

### 1. Future VUS resolution contains learnable predictive signal

XGBoost substantially outperformed the Logistic Regression baseline.

```text
Logistic PR-AUC: 0.0171
XGBoost PR-AUC:  0.0296
```

---

### 2. ClinVar evaluation history was the strongest predictive signal

`evaluation_age_days` consistently dominated SHAP feature importance in both the full and population-blind models.

---

### 3. Explicit population-frequency features were not required for prediction

The population-blind XGBoost achieved:

```text
ROC-AUC = 0.7406
PR-AUC  = 0.0346
```

which slightly exceeded the full model.

---

### 4. Population-derived features strongly influenced the full model

Allele-number and allele-frequency features were among the most important full-model predictors.

However, their importance may partly reflect population-data availability rather than variant biology.

---

### 5. SAS/NFE SHAP differences became smaller in the population-blind model

When explicit population features were removed, the remaining differences were mainly related to:

- review status
- evaluation age
- multi-gene annotation

---

### 6. Calibration was essential

The class-weighted XGBoost was severely overconfident before calibration.

Platt scaling reduced the Brier score from:

```text
0.1871 → 0.0110
```

while preserving discrimination.

---

## End-to-End Pipeline

```text
ClinVar February 2024
        |
        v
Extract T1 VUS
        |
        v
Variant QC + GRCh38 filtering
        |
        v
ClinVar July 2025 linkage
        |
        v
Construct future-resolution target
        |
        v
Normalize chr22 variants
        |
        v
Match against gnomAD v4.1
        |
        v
Extract SAS / NFE AF, AC, AN
        |
        v
Fisher exact tests
        |
        v
BH-FDR enrichment labels
        |
        v
Feature engineering
        |
        v
Stratified 80/20 split
        |
        +---------------------------+
        |                           |
        v                           v
Logistic Regression             XGBoost
Baseline                           |
                                   v
                           5-fold CV tuning
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
               Full XGBoost             Population-Blind
                     |                           |
                     v                           v
                   SHAP                        SHAP
                     |                           |
                     +-------------+-------------+
                                   |
                                   v
                         SAS vs NFE audit
                                   |
                                   v
                         Probability calibration
                                   |
                                   v
                             Final analysis
```

---

## Repository Structure

```text
genomic-variant-equity/
│
├── data/
│   ├── raw/
│   │   ├── clinvar/
│   │   └── gnomad/
│   │
│   ├── interim/
│   │
│   └── processed/
│       ├── gnomad/
│       └── ml/
│
├── src/
│   ├── clinvar/
│   │
│   ├── gnomad/
│   │   ├── run_chromosome_pipeline.py
│   │   └── create_chr22_enrichment_labels.py
│   │
│   └── modeling/
│       ├── build_ml_dataset.py
│       ├── make_train_test_split.py
│       ├── train_logistic_baseline.py
│       ├── train_xgboost.py
│       ├── tune_xgboost.py
│       ├── evaluate_tuned_xgboost.py
│       ├── select_threshold.py
│       ├── run_shap.py
│       ├── compare_sas_nfe_shap.py
│       ├── train_population_blind_xgboost.py
│       ├── compare_population_blind_sas_nfe_shap.py
│       ├── evaluate_calibration.py
│       ├── summarize_final_results.py
│       └── make_final_figures.py
│
└── results/
    ├── shap/
    ├── population_blind/
    └── final/
```

---

## Generated Results

### Final Tables

```text
results/final/model_comparison.csv
results/final/calibration_comparison.csv
results/final/sas_nfe_summary.csv
results/final/shap_summary.csv
```

### Final Figures

```text
results/final/figures/model_roc_auc.png
results/final/figures/model_pr_auc.png
results/final/figures/calibration_brier_score.png
results/final/figures/sas_nfe_resolution_rate.png
```

### SHAP Outputs

```text
results/shap/shap_bar_named.png
results/shap/shap_beeswarm_named.png
results/shap/shap_feature_importance.csv
results/shap/shap_values_test.csv
```

### Population-Blind Outputs

```text
results/population_blind/population_blind_shap_values_test.csv
results/population_blind/population_blind_feature_importance.csv
results/population_blind/subgroup/population_blind_sas_vs_nfe_shap.csv
```

---

## Methodological Safeguards

Several design choices were used to reduce leakage and overstatement.

### Temporal Leakage Prevention

All predictors came from the T1 snapshot.

T2 information was used only for outcome construction.

### Train-Test Leakage Prevention

The test set was kept untouched during hyperparameter tuning.

Preprocessing parameters were fitted on training data.

### Gene-Frequency Leakage Prevention

A gene-frequency feature initially calculated before splitting was removed from final model analyses.

### Threshold Selection

The decision threshold was selected using out-of-fold training predictions rather than test-set performance.

### Population-Audit Interpretation

SAS/NFE groups describe variants enriched in gnomAD populations.

They do not represent individual patient ancestry.

SHAP differences were treated as exploratory model-behavior differences, not as evidence of causal inequity.

---

## Limitations

This project is a **chromosome 22 proof-of-concept**, not a genome-wide clinical prediction system.

Important limitations include:

- analysis is restricted to chromosome 22
- only 286 variants resolved in the final cohort
- only 57 positive cases were present in the held-out test set
- subgroup resolved counts were particularly small
- no external independent validation dataset was used
- gnomAD SAS/NFE frequencies are population-level proxies and not patient ancestry
- ClinVar reclassification reflects evidence accumulation and submission practices, not biology alone
- allele-number variables may capture data availability and sequencing coverage
- SHAP measures model attribution rather than causality
- subgroup SHAP differences do not establish algorithmic bias or clinical inequity
- small differences between model performances should be interpreted cautiously
- the system is not intended to determine variant pathogenicity
- the model is not intended for clinical use

---

## Interpretation

This project predicts:

> **Which currently uncertain ClinVar variants are more likely to receive a resolved interpretation in the future?**

It should therefore be viewed as an experimental:

**VUS prioritization + explainable ML + population-aware model auditing framework**

rather than a variant-classification or clinical decision-support system.

---

## Technologies Used

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- SHAP
- statsmodels
- matplotlib
- bcftools
- samtools
- pysam
- ClinVar
- gnomAD v4.1

---

## Project Status

**Chromosome 22 proof-of-concept complete.**

- [x] Historical ClinVar T1 cohort construction
- [x] Later ClinVar T2 linkage
- [x] Longitudinal resolution-label construction
- [x] GRCh38 variant QC
- [x] Variant normalization
- [x] gnomAD population-frequency integration
- [x] SAS/NFE population comparison
- [x] Fisher exact testing
- [x] BH-FDR correction
- [x] Population-enrichment labeling
- [x] ML feature engineering
- [x] Leakage-safe train-test split
- [x] Logistic Regression baseline
- [x] XGBoost baseline
- [x] Cross-validated hyperparameter tuning
- [x] Threshold analysis
- [x] SHAP explainability
- [x] SAS/NFE SHAP audit
- [x] Population-blind sensitivity model
- [x] Population-blind subgroup SHAP analysis
- [x] Platt calibration
- [x] Isotonic calibration
- [x] Final model comparison
- [x] Final figures and result tables

---

## Future Work

Potential extensions include:

- genome-wide analysis across all autosomes
- external validation on an independent temporal ClinVar cohort
- additional functional evidence sources
- protein-level or sequence-based embeddings
- literature-derived evidence
- gene-level biological annotations
- conformal prediction for calibrated uncertainty
- prospective VUS-prioritization evaluation
- stronger statistical analysis of subgroup performance with larger positive counts

---

## Disclaimer

This project is for **research and educational purposes only**.

It is **not intended for clinical diagnosis, medical decision-making, or direct interpretation of genetic variants**.