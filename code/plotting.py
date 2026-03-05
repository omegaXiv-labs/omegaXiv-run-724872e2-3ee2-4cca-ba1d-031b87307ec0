from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")


def plot_multi_panel_results(summary_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    panel_settings = ["A", "B", "C"]
    for idx, setting in enumerate(panel_settings):
        ax = axes[idx]
        sdf = summary_df[summary_df["setting"] == setting].copy()
        sdf = sdf.sort_values("target_mse_mean", ascending=True)
        x = range(len(sdf))
        y = sdf["target_mse_mean"]
        yerr_low = y - sdf["target_mse_ci_low"]
        yerr_high = sdf["target_mse_ci_high"] - y

        ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            capsize=4,
            label="Mean ± 95% CI",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(sdf["method"], rotation=35, ha="right")
        ax.set_ylabel("Target MSE (squared error units)")
        ax.set_xlabel("Method")
        ax.set_title(f"Setting {setting}")
        ax.legend(loc="best")

    fig.suptitle(
        "Caption: Cross-setting comparison of IWLS source selection methods with 95% CI across five seeds",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def plot_stability_tradeoff(results_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax0 = axes[0]
    sns.scatterplot(
        data=results_df,
        x="ess",
        y="target_mse",
        hue="method",
        style="setting",
        ax=ax0,
    )
    ax0.set_xlabel("Effective sample size (ESS)")
    ax0.set_ylabel("Target MSE (squared error units)")
    ax0.set_title("ESS vs Target Error")
    ax0.legend(loc="best")

    ax1 = axes[1]
    sns.scatterplot(
        data=results_df,
        x="condition_number",
        y="target_mse",
        hue="method",
        style="setting",
        ax=ax1,
    )
    ax1.set_xlabel("Condition number κ(XᵀWX)")
    ax1.set_ylabel("Target MSE (squared error units)")
    ax1.set_title("Conditioning vs Target Error")
    ax1.legend(loc="best")

    fig.suptitle(
        "Caption: Stability diagnostics versus target performance for all settings and baselines",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def plot_ablation_vs_pooled(ablation_df: pd.DataFrame, out_pdf: Path) -> None:
    if ablation_df.empty:
        return

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax0 = axes[0]
    sns.scatterplot(
        data=ablation_df,
        x="pooled_gap_mean",
        y="holm_adjusted_p",
        hue="ratio_estimator",
        style="clipping",
        size="gamma",
        ax=ax0,
    )
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax0.axhline(0.05, color="black", linestyle=":", linewidth=1.0)
    ax0.set_xlabel("Mean MSE gap: stability - pooled (lower is better)")
    ax0.set_ylabel("Holm-adjusted p-value")
    ax0.set_title("Ablation significance vs pooled IWLS")

    ax1 = axes[1]
    top = ablation_df.nsmallest(12, columns=["holm_adjusted_p", "pooled_gap_mean"]).copy()
    top["label"] = (
        "a="
        + top["alpha"].astype(str)
        + ", b="
        + top["beta"].astype(str)
        + ", g="
        + top["gamma"].astype(str)
        + ", clip="
        + top["clipping"].astype(str)
        + ", est="
        + top["ratio_estimator"].astype(str)
    )
    sns.barplot(data=top, x="pooled_gap_mean", y="label", hue="adaptive_gamma", ax=ax1)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Mean MSE gap: stability - pooled")
    ax1.set_ylabel("Top ablation configurations")
    ax1.set_title("Best pooled-gap configurations")

    fig.suptitle("Caption: Pooled-IWLS-focused ablation over stability calibration controls", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
