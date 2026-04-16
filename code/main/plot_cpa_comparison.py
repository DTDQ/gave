#!/usr/bin/env python3
"""
plot_cpa_comparison.py - 对比多组实验与真实数据的 CPA 表现

默认读取 4 组实验和 real_details.csv，生成一张包含：
1. 平均 CPA 柱状图
2. 按 real CPA 排序后的逐任务 CPA 曲线图
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUTS = {
    "real": "experiments/exp_score_original/results/test_results_500/real_details.csv",
    "main": "experiments/exp_score_original/results/test_results_500/details_20260416_075101/best_model.pt_details.csv",
    "exp": "experiments/exp_score_exp_penalty/results/test_results_500/details_20260416_002229/best_model.pt_details.csv",
    "power": "experiments/exp_score_power_penalty/results/test_results_500/details_20260416_013637/best_model.pt_details.csv",
    "cliff": "experiments/exp_score_cliff_penalty/results/test_results_500/details_20260416_024942/best_model.pt_details.csv",
}

DISPLAY_NAMES = {
    "real": "Real",
    "main": "Main",
    "exp": "Exp",
    "power": "Power",
    "cliff": "Cliff",
}

COLORS = {
    "real": "#1f1f1f",
    "main": "#4c78a8",
    "exp": "#59a14f",
    "power": "#e15759",
    "cliff": "#f28e2b",
}


def load_cpa_frame(path: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return (
        df[["ad_id", "period", "CPA"]]
        .rename(columns={"CPA": f"CPA_{key}"})
        .copy()
    )


def build_merged_frame(inputs: dict[str, str]) -> pd.DataFrame:
    real_df = load_cpa_frame(Path(inputs["real"]), "real")
    merged = real_df.copy()

    for key, path in inputs.items():
        if key == "real":
            continue
        merged = merged.merge(
            load_cpa_frame(Path(path), key),
            on=["ad_id", "period"],
            how="inner",
        )

    merged = merged.sort_values("CPA_real").reset_index(drop=True)
    return merged


def plot_cpa_comparison(merged: pd.DataFrame, output_path: Path) -> None:
    keys = ["real", "main", "exp", "power", "cliff"]
    avg_cpa = [merged[f"CPA_{key}"].mean() for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CPA Comparison Across Experiments", fontsize=16, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(
        [DISPLAY_NAMES[key] for key in keys],
        avg_cpa,
        color=[COLORS[key] for key in keys],
        alpha=0.9,
    )
    ax.set_title("Average CPA")
    ax.set_ylabel("CPA")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, avg_cpa):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    x = range(len(merged))
    for key in keys:
        ax.plot(
            x,
            merged[f"CPA_{key}"],
            label=DISPLAY_NAMES[key],
            color=COLORS[key],
            linewidth=1.5 if key == "real" else 1.1,
            alpha=0.95 if key == "real" else 0.85,
        )
    ax.set_title("Per-Trajectory CPA (Sorted by Real CPA)")
    ax.set_xlabel("Trajectory Rank")
    ax.set_ylabel("CPA")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"CPA 对比图已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制多实验 CPA 对比图")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/cpa_comparison_4exp_vs_real.png",
        help="输出图片路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = build_merged_frame(DEFAULT_INPUTS)
    plot_cpa_comparison(merged, Path(args.output))


if __name__ == "__main__":
    main()
