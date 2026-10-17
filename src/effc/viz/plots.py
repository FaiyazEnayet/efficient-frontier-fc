"""Plots: frontiers, squads, backtests. Best-effort, no interactivity."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_frontier(frontier: pd.DataFrame, outdir: str, budgets_m: np.ndarray | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = frontier["budget"] / 1e6
    ax.plot(x, frontier["exp_vaep_per90"], marker="o", ms=4)
    ax.set_xlabel("wage budget (EURm / week)")
    ax.set_ylabel("expected squad VAEP/90")
    ax.set_title("Efficient frontier: max expected value per wage budget")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "frontier.png"), dpi=120)
    plt.close(fig)


def plot_backtest(bt: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bt))
    w = 0.15
    for i, col in enumerate(["optimizer", "value", "random_mean", "oracle"]):
        ax.bar(x + (i - 1.5) * w, bt[col], width=w, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b/1e6:.1f}m" for b in bt["budget"]])
    ax.set_xlabel("wage budget (EURm / week)")
    ax.set_ylabel("realized second-half squad VAEP/90")
    ax.set_title("Backtest: first-half squads, second-half reality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "backtest.png"), dpi=120)
    plt.close(fig)


def plot_vaep_leaders(ratings: pd.DataFrame, outdir: str, n: int = 20):
    top = ratings.nlargest(n, "vaep_per90")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["player"][::-1], top["vaep_per90"][::-1])
    ax.set_xlabel("shrunk VAEP/90")
    ax.set_title(f"Top {n} players by VAEP/90 (Bundesliga 2023/24)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "vaep_leaders.png"), dpi=120)
    plt.close(fig)
