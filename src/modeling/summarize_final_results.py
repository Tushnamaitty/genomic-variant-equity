from pathlib import Path
import pandas as pd

OUTDIR = Path("results/final")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Main model comparison
# -----------------------------

model_results = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "roc_auc": 0.6490,
        "pr_auc": 0.0171,
        "precision_at_0.5": 0.0173,
        "recall_at_0.5": 0.6842,
        "f1_at_0.5": 0.0338,
    },
    {
        "model": "XGBoost Full",
        "roc_auc": 0.7348,
        "pr_auc": 0.0296,
        "precision_at_0.5": 0.0245,
        "recall_at_0.5": 0.5965,
        "f1_at_0.5": 0.0472,
    },
    {
        "model": "XGBoost Population-Blind",
        "roc_auc": 0.7406,
        "pr_auc": 0.0346,
        "precision_at_0.5": None,
        "recall_at_0.5": None,
        "f1_at_0.5": None,
    },
])

model_results.to_csv(
    OUTDIR / "model_comparison.csv",
    index=False
)

# -----------------------------
# 2. Calibration comparison
# -----------------------------

calibration = pd.DataFrame([
    {
        "method": "Uncalibrated",
        "roc_auc": 0.7406,
        "pr_auc": 0.0346,
        "brier_score": 0.187140,
    },
    {
        "method": "Platt",
        "roc_auc": 0.7413,
        "pr_auc": 0.0333,
        "brier_score": 0.011037,
    },
    {
        "method": "Isotonic",
        "roc_auc": 0.7377,
        "pr_auc": 0.0358,
        "brier_score": 0.011093,
    },
])

calibration.to_csv(
    OUTDIR / "calibration_comparison.csv",
    index=False
)

# -----------------------------
# 3. SAS/NFE cohort summary
# -----------------------------

subgroups = pd.DataFrame([
    {
        "group": "SAS_enriched",
        "n_variants": 1709,
        "n_resolved": 16,
        "resolution_percent": 0.9362,
    },
    {
        "group": "NFE_enriched",
        "n_variants": 1408,
        "n_resolved": 9,
        "resolution_percent": 0.6392,
    },
])

subgroups.to_csv(
    OUTDIR / "sas_nfe_summary.csv",
    index=False
)

# -----------------------------
# 4. Key SHAP findings
# -----------------------------

shap_summary = pd.DataFrame([
    {
        "finding": "Strongest overall predictor",
        "result": "evaluation_age_days",
    },
    {
        "finding": "Important full-model population features",
        "result": "AN_joint_sas, AN_joint_nfe, AF differences",
    },
    {
        "finding": "Important non-population feature",
        "result": "ClinVar review_status",
    },
    {
        "finding": "Population-blind strongest predictor",
        "result": "evaluation_age_days",
    },
    {
        "finding": "Largest population-blind SAS/NFE SHAP difference",
        "result": "review_status",
    },
])

shap_summary.to_csv(
    OUTDIR / "shap_summary.csv",
    index=False
)

# -----------------------------
# 5. Print final summary
# -----------------------------

print("=== MODEL COMPARISON ===")
print(model_results.to_string(index=False))

print("\n=== CALIBRATION ===")
print(calibration.to_string(index=False))

print("\n=== SAS/NFE SUMMARY ===")
print(subgroups.to_string(index=False))

print("\n=== KEY SHAP FINDINGS ===")
print(shap_summary.to_string(index=False))

print("\nSaved to:", OUTDIR)