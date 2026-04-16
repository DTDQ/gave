#!/usr/bin/env python3
"""
plot_loss.py - 绘制多组实验与真实数据的 CPA 散点图
用法: python plot_loss.py [--output <output_path>]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


DEFAULT_INPUTS = {
    "real": "experiments/exp_score_original/results/test_results_500/real_details.csv",
    "main": "experiments/exp_score_original/results/test_results_500/details_20260416_075101/best_model.pt_details.csv",
    "exp": "experiments/exp_score_exp_penalty/results/test_results_500/details_20260416_002229/best_model.pt_details.csv",
    "power": "experiments/exp_score_power_penalty/results/test_results_500/details_20260416_013637/best_model.pt_details.csv",
    "cliff": "experiments/exp_score_cliff_penalty/results/test_results_500/details_20260416_024942/best_model.pt_details.csv",
}

DISPLAY_NAMES = {
    "real": "Real",
    "main": "Main (Original)",
    "exp": "Exp Penalty",
    "power": "Power Penalty",
    "cliff": "Cliff Penalty",
}

COLORS = {
    "real": "#1f1f1f",
    "main": "#4c78a8",
    "exp": "#59a14f",
    "power": "#e15759",
    "cliff": "#f28e2b",
}

MARKERS = {
    "real": "o",
    "main": "s",
    "exp": "^",
    "power": "D",
    "cliff": "v",
}


def load_cpa(path: str, key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["ad_id", "period", "CPA"]].rename(columns={"CPA": f"CPA_{key}"})


def build_merged(inputs: dict[str, str]) -> pd.DataFrame:
    real_df = load_cpa(inputs["real"], "real")
    merged = real_df.copy()
    for key, path in inputs.items():
        if key == "real":
            continue
        merged = merged.merge(load_cpa(path, key), on=["ad_id", "period"], how="inner")
    merged = merged.sort_values("CPA_real").reset_index(drop=True)
    return merged


def plot_cpa_scatter(merged: pd.DataFrame, output_path: Path) -> None:
    keys = ["real", "main", "exp", "power", "cliff"]
    n = len(merged)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle("CPA Distribution Across Test Trajectories", fontsize=16, fontweight="bold")

    for key in keys:
        cpa = merged[f"CPA_{key}"].values
        ax.scatter(
            x, cpa,
            label=f"{DISPLAY_NAMES[key]} (mean={cpa.mean():.2f})",
            color=COLORS[key],
            marker=MARKERS[key],
            s=18,
            alpha=0.6,
            edgecolors="none",
        )

    ax.set_xlabel("Trajectory Index (sorted by Real CPA)", fontsize=12)
    ax.set_ylabel("CPA", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, markerscale=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"CPA 散点图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制多实验 CPA 散点分布图")
    parser.add_argument("--output", type=str, default="experiments/cpa_scatter_distribution.png",
                        help="输出图片路径")
    args = parser.parse_args()

    merged = build_merged(DEFAULT_INPUTS)
    print(f"共 {len(merged)} 条轨迹")
    plot_cpa_scatter(merged, Path(args.output))


if __name__ == "__main__":
    main()
