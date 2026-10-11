"""Generate docs/demo.gif: an animated preview of the dashboard.

Programmatic matplotlib animation built from the repo's own demo tables
(see scripts/make_demo.py): the efficient frontier draws itself, a marker
sweeps the budget grid, and the best XI at each budget is listed beside
it. Nothing is faked or downloaded.

Run from the repo root:  python scripts/make_gif.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.demo import load_demo  # noqa: E402
from effc.portfolio.portfolio import select_xi  # noqa: E402

OUT = os.path.join("docs", "demo.gif")


def main() -> None:
    data = load_demo()
    frontier: pd.DataFrame = data["frontier"]
    squads: pd.DataFrame = data["squads"]
    formation: str = data["meta"].get("formation", "4-3-3")

    budgets = frontier["budget"].tolist()
    xi_by_budget = {}
    for b in budgets:
        squad = squads[squads["budget"] == b].copy()
        xi_by_budget[b] = select_xi(squad, formation=formation).sort_values(
            "vaep_per90", ascending=False
        )

    fig, (ax_curve, ax_xi) = plt.subplots(1, 2, figsize=(11, 5.2),
                                          gridspec_kw={"width_ratios": [1.25, 1]})
    x = frontier["budget"] / 1e6
    y = frontier["exp_vaep_per90"]
    ax_curve.plot(x, y, color="#1f77b4", lw=2)
    dot, = ax_curve.plot([], [], "o", color="#d62728", ms=10)
    ax_curve.set_xlabel("wage budget (EURm / week)")
    ax_curve.set_ylabel("expected squad VAEP/90")
    ax_curve.set_title("Efficient frontier")
    ax_curve.grid(alpha=0.3)
    budget_text = ax_curve.text(
        0.04, 0.94, "", transform=ax_curve.transAxes, fontsize=12,
        fontweight="bold", va="top",
    )

    ax_xi.axis("off")
    xi_title = ax_xi.text(0.02, 0.96, "", fontsize=12, fontweight="bold",
                          transform=ax_xi.transAxes, va="top")
    xi_body = ax_xi.text(0.02, 0.88, "", fontsize=9.5, family="monospace",
                         transform=ax_xi.transAxes, va="top", linespacing=1.6)

    def frame(i: int):
        b = budgets[i]
        v = float(frontier.loc[frontier["budget"] == b, "exp_vaep_per90"].iloc[0])
        dot.set_data([b / 1e6], [v])
        budget_text.set_text(f"Budget EUR {b:,.0f}/wk\nSquad VAEP/90: {v:.2f}")
        xi = xi_by_budget[b]
        xi_title.set_text(f"Best XI ({formation})")
        lines = [
            f"{r['player'][:22]:22s} {r['pos_group']:3s} {r['vaep_per90']:.2f}"
            for _, r in xi.iterrows()
        ]
        xi_body.set_text("\n".join(lines))
        return dot, budget_text, xi_title, xi_body

    anim = FuncAnimation(fig, frame, frames=len(budgets), blit=False)
    os.makedirs("docs", exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
