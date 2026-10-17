"""Build docs/dashboard.html: one self-contained interactive Plotly page.

Reads demo/*.csv (see scripts/make_demo.py) and writes a single HTML file
with the plotly.js bundle inlined, so it opens straight off disk or from
GitHub Pages with no server and no install.

Sections:
  1. Frontier explorer: budget slider sweeping the efficient frontier.
     Curve + marker + 23-man squad table + best XI table update per step.
  2. Top rated players: bar chart with a position-group dropdown.
  3. Backtest: grouped bars across selection rules.

Run from the repo root:  python scripts/make_dashboard.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, plot
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.demo import load_demo  # noqa: E402
from effc.portfolio.portfolio import select_xi  # noqa: E402

OUT = os.path.join("docs", "dashboard.html")
POS_COLORS = {"GK": "#636EFA", "DEF": "#00CC96", "MID": "#AB63FA", "FWD": "#EF553B"}
RULE_LABELS = {
    "optimizer": "optimizer (MILP)",
    "value": "value per euro",
    "random_mean": "random",
    "oracle": "oracle",
}


def eur(b: float) -> str:
    return f"EUR {b:,.0f}"


def budget_label(b: float) -> str:
    return f"EUR {b / 1e6:.2f}M / week"


def _table_trace(df: pd.DataFrame, title: str) -> go.Table:
    vals = [
        df["player"].tolist(),
        df["team"].tolist(),
        df["pos_group"].tolist(),
        [f"{v:.2f}" for v in df["vaep_per90"]],
        [f"{w:,.0f}" for w in df["wage_eur"]],
    ]
    return go.Table(
        header=dict(
            values=["Player", "Team", "Pos", "VAEP/90", "Wage/wk"],
            fill_color="#1f2937", font_color="white", align="left",
        ),
        cells=dict(values=vals, fill_color="#f3f4f6", align="left", height=22),
        name=title,
    )


def frontier_figure(data: dict) -> go.Figure:
    frontier: pd.DataFrame = data["frontier"]
    squads: pd.DataFrame = data["squads"]
    formation: str = data["meta"].get("formation", "4-3-3")

    budgets = frontier["budget"].tolist()
    # best XI per grid budget (fast MILP on 23 players)
    xi_by_budget = {}
    for b in budgets:
        squad = squads[squads["budget"] == b].copy()
        xi = select_xi(squad, formation=formation).sort_values(
            "vaep_per90", ascending=False
        )
        xi_by_budget[b] = xi

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "table"}, {"type": "table"}]],
        subplot_titles=(
            "Efficient frontier: max expected squad VAEP/90 per wage budget",
            "23-man squad", f"Best XI ({formation})",
        ),
        vertical_spacing=0.12, row_heights=[0.45, 0.55],
    )
    fig.add_trace(
        go.Scatter(
            x=frontier["budget"] / 1e6, y=frontier["exp_vaep_per90"],
            mode="lines+markers", name="frontier",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="EUR %{x:.2f}M/wk<br>VAEP/90: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    b0 = budgets[0]
    v0 = float(frontier.loc[frontier["budget"] == b0, "exp_vaep_per90"].iloc[0])
    fig.add_trace(
        go.Scatter(
            x=[b0 / 1e6], y=[v0], mode="markers", name="selected budget",
            marker=dict(color="#d62728", size=12),
            hovertemplate="%{x:.2f}M/wk: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    squad0 = squads[squads["budget"] == b0].sort_values("vaep_per90", ascending=False)
    fig.add_trace(_table_trace(squad0, "squad"), row=2, col=1)
    fig.add_trace(_table_trace(xi_by_budget[b0], "xi"), row=2, col=2)

    frames = []
    for b in budgets:
        v = float(frontier.loc[frontier["budget"] == b, "exp_vaep_per90"].iloc[0])
        sq = squads[squads["budget"] == b].sort_values("vaep_per90", ascending=False)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[b / 1e6], y=[v], mode="markers"),
                    _table_trace(sq, "squad"),
                    _table_trace(xi_by_budget[b], "xi"),
                ],
                traces=[1, 2, 3],
                name=budget_label(b),
            )
        )
    fig.frames = frames

    steps = [
        dict(
            label=budget_label(b), method="animate",
            args=[[budget_label(b)],
                  {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                   "transition": {"duration": 0}}],
        )
        for b in budgets
    ]
    fig.update_layout(
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "Budget: "},
                      pad={"t": 40})],
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=1.12,
            buttons=[
                dict(label="Play sweep", method="animate",
                     args=[None, {"frame": {"duration": 700, "redraw": True},
                                 "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"mode": "immediate"}]),
            ],
        )],
        height=900, margin=dict(t=110),
        xaxis_title="wage budget (EURm / week)",
        yaxis_title="expected squad VAEP/90",
        showlegend=False,
    )
    return fig


def ratings_figure(data: dict) -> go.Figure:
    players: pd.DataFrame = data["players"]
    fig = go.Figure()
    groups = ["All", "GK", "DEF", "MID", "FWD"]
    for i, g in enumerate(groups):
        pool = players if g == "All" else players[players["pos_group"] == g]
        top = pool.nlargest(15, "vaep_per90").iloc[::-1]
        fig.add_trace(
            go.Bar(
                x=top["vaep_per90"], y=top["player"], orientation="h",
                visible=(i == 0), name=g,
                marker_color=[POS_COLORS[p] for p in top["pos_group"]],
                customdata=list(
                    zip(top["team"], top["pos_group"], top["wage_eur"], top["minutes"])
                ),
                hovertemplate=(
                    "%{y} (%{customdata[1]}, %{customdata[0]})<br>"
                    "VAEP/90: %{x:.2f}<br>Wage: EUR %{customdata[2]:,.0f}/wk<br>"
                    "Minutes: %{customdata[3]:,.0f}<extra></extra>"
                ),
            )
        )
    buttons = [
        dict(label=g, method="update",
             args=[{"visible": [j == i for j in range(len(groups))]},
                   {"title": f"Top 15 rated players ({g})"}])
        for i, g in enumerate(groups)
    ]
    fig.update_layout(
        title="Top 15 rated players (All)",
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.0, y=1.12,
                          showactive=True)],
        xaxis_title="shrunk VAEP/90", height=560, margin=dict(t=100),
    )
    return fig


def backtest_figure(data: dict) -> go.Figure:
    bt: pd.DataFrame = data["backtest"].copy()
    bt["blabel"] = (bt["budget"] / 1e6).map(lambda b: f"EUR {b:.1f}M")
    fig = go.Figure()
    for rule in ["optimizer", "value", "random_mean", "oracle"]:
        sub = bt[bt["rule"] == rule].sort_values("budget")
        fig.add_trace(
            go.Bar(
                x=sub["blabel"], y=sub["score"], name=RULE_LABELS[rule],
                hovertemplate="%{x}<br>" + RULE_LABELS[rule] + ": %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Backtest: first-half squads scored on the second half",
        barmode="group", xaxis_title="wage budget / week",
        yaxis_title="realized second-half squad VAEP/90", height=480,
        annotations=[dict(
            x="EUR 0.7M", y=14.4, text="value beats optimizer at 700k",
            showarrow=True, arrowhead=2, ax=0, ay=-40,
        )],
    )
    return fig


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Efficient Frontier FC: interactive dashboard</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px; color: #111827; }}
h1 {{ margin-bottom: 4px; }} .sub {{ color: #4b5563; margin-top: 0; }}
section {{ margin: 40px 0; }} footer {{ color: #6b7280; font-size: 0.85em;
       border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 48px; }}
.facts li {{ margin: 4px 0; }}
</style>
<script>{plotlyjs}</script>
</head>
<body>
<h1>Efficient Frontier FC</h1>
<p class="sub">Squad building as portfolio optimization. Interactive companion
to <a href="https://github.com/FaiyazEnayet/efficient-frontier-fc">the repo</a>.</p>
<ul class="facts">
<li>VAEP player ratings from 196 StatsBomb matches (Bundesliga 2023/24 and 2015/16).</li>
<li>Wages: EA FC 24 weekly wages in EUR.</li>
<li>Drag the budget slider: the squad MILP optimum, the 23-man squad and the best XI update at each step.</li>
</ul>
<section><h2>Frontier explorer</h2>{div_frontier}</section>
<section><h2>Top rated players</h2>{div_ratings}</section>
<section><h2>Backtest</h2>
<p>Every rule respects the same position quotas and the same wage budget; only the selection rule differs.
The oracle re-runs the optimizer on second-half ratings, so it has perfect foresight of the ratings, but it maximizes expected VAEP under the model, not the realized score. It is a strong reference, not a guaranteed ceiling.</p>
{div_backtest}</section>
<footer>Generated by <code>scripts/make_dashboard.py</code> from the pipeline's
demo tables (<code>demo/</code>). Numbers match
<code>runs/bundesliga_2seasons/metrics.json</code>.</footer>
</body>
</html>
"""


def main() -> None:
    data = load_demo()
    div_frontier = plot(frontier_figure(data), output_type="div", include_plotlyjs=False)
    div_ratings = plot(ratings_figure(data), output_type="div", include_plotlyjs=False)
    div_backtest = plot(backtest_figure(data), output_type="div", include_plotlyjs=False)
    html = PAGE.format(
        plotlyjs=get_plotlyjs(), div_frontier=div_frontier,
        div_ratings=div_ratings, div_backtest=div_backtest,
    )
    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
