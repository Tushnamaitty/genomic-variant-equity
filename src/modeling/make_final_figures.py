from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/final")
FIG_DIR = Path("results/final/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)


def model_comparison():

    df = pd.read_csv(
        RESULTS_DIR / "model_comparison.csv"
    )

    # ROC-AUC
    plt.figure(figsize=(8, 5))

    plt.bar(
        df["model"],
        df["roc_auc"]
    )

    plt.ylabel("ROC-AUC")
    plt.title("Model Comparison: ROC-AUC")
    plt.ylim(0.5, 0.8)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "model_roc_auc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # PR-AUC
    plt.figure(figsize=(8, 5))

    plt.bar(
        df["model"],
        df["pr_auc"]
    )

    plt.ylabel("PR-AUC")
    plt.title("Model Comparison: PR-AUC")

    # Random baseline = positive prevalence
    plt.axhline(
        y=0.0113,
        linestyle="--",
        label="Random baseline (1.13%)"
    )

    plt.legend()

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "model_pr_auc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calibration_plot():

    df = pd.read_csv(
        RESULTS_DIR /
        "calibration_comparison.csv"
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        df["method"],
        df["brier_score"]
    )

    plt.ylabel("Brier Score (lower is better)")
    plt.title("Probability Calibration Comparison")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "calibration_brier_score.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def subgroup_resolution():

    df = pd.read_csv(
        RESULTS_DIR /
        "sas_nfe_summary.csv"
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        df["group"],
        df["resolution_percent"]
    )

    plt.ylabel("VUS Resolution Rate (%)")
    plt.title(
        "Observed VUS Resolution by Population-Enrichment Group"
    )

    for i, row in df.iterrows():

        plt.text(
            i,
            row["resolution_percent"] + 0.02,
            f'{row["resolution_percent"]:.2f}%\n'
            f'n={int(row["n_variants"])}',
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "sas_nfe_resolution_rate.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    model_comparison()
    calibration_plot()
    subgroup_resolution()

    print("Final figures created:")
    print(FIG_DIR / "model_roc_auc.png")
    print(FIG_DIR / "model_pr_auc.png")
    print(FIG_DIR / "calibration_brier_score.png")
    print(FIG_DIR / "sas_nfe_resolution_rate.png")

    print("\nExisting SHAP figures:")
    print("results/shap/shap_bar_named.png")
    print("results/shap/shap_beeswarm_named.png")


if __name__ == "__main__":
    main()