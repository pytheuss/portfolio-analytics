"""Plotly charts for the app.

Colour follows the entity: the portfolio is always blue and the benchmark always
orange, on every chart. Single-series charts use the portfolio blue. Text uses
neutral ink, never the series colour. Light and dark modes get their own shades.
"""

import pandas as pd
import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

PALETTE = {
    "light": {"portfolio": "#2a78d6", "benchmark": "#eb6834", "ink": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "critical": "#d03b3b"},
    "dark": {"portfolio": "#3987e5", "benchmark": "#d95926", "ink": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "critical": "#d03b3b"},
}


def colors(mode: str) -> dict:
    return PALETTE["dark" if mode == "dark" else "light"]


def _style(fig: go.Figure, c: dict, y_format: str, height: int = 340) -> go.Figure:
    """Shared look: transparent surface, hairline grid, legend above the plot."""
    fig.update_layout(
        template="none", height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=c["ink"], size=13),
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    # automargin grows the margins to fit tick labels (the blank template leaves it off).
    fig.update_yaxes(tickformat=y_format, gridcolor=c["grid"], zeroline=False, automargin=True,
                     linecolor=c["axis"], tickfont=dict(color=c["muted"]))
    fig.update_xaxes(showgrid=False, linecolor=c["axis"], tickfont=dict(color=c["muted"]),
                     automargin=True)
    return fig


def _end_label(fig: go.Figure, series: pd.Series, text: str, c: dict) -> None:
    """Direct label at the last point of a line, in neutral ink."""
    fig.add_annotation(x=series.index[-1], y=series.iloc[-1], text=text, showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=c["ink"], size=12))


def growth_chart(growth: pd.DataFrame, mode: str) -> go.Figure:
    """Cumulative time-weighted return: portfolio vs benchmark."""
    c = colors(mode)
    fig = go.Figure()
    for name, label in [("portfolio", "Portfolio"), ("benchmark", "Benchmark")]:
        cumulative = growth[name] - 1
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name=label, mode="lines",
                                 line=dict(color=c[name], width=2),
                                 hovertemplate="%{y:+.1%}<extra>" + label + "</extra>"))
        _end_label(fig, cumulative, f"{label} {cumulative.iloc[-1]:+.1%}", c)
    _style(fig, c, "+.0%")
    # Room on the right for the end labels, which sit outside the plot area.
    fig.update_layout(hovermode="x unified", margin=dict(r=130))
    fig.update_xaxes(range=[growth.index[0], growth.index[-1]])
    return fig


def drawdown_chart(portfolio: pd.Series, benchmark: pd.Series, mode: str) -> go.Figure:
    """Distance below the previous high, portfolio (shaded) vs benchmark."""
    c = colors(mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio, name="Portfolio", mode="lines",
                             line=dict(color=c["portfolio"], width=2), fill="tozeroy",
                             fillcolor=_alpha(c["portfolio"], 0.15),
                             hovertemplate="%{y:.1%}<extra>Portfolio</extra>"))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Benchmark", mode="lines",
                             line=dict(color=c["benchmark"], width=2),
                             hovertemplate="%{y:.1%}<extra>Benchmark</extra>"))
    fig.update_layout(hovermode="x unified")
    return _style(fig, c, ".0%", height=280)


def monthly_chart(monthly: pd.DataFrame, mode: str) -> go.Figure:
    """Monthly returns, portfolio and benchmark side by side."""
    c = colors(mode)
    months = [str(m) for m in monthly.index]
    fig = go.Figure()
    for name, label in [("portfolio", "Portfolio"), ("benchmark", "Benchmark")]:
        fig.add_trace(go.Bar(x=months, y=monthly[name], name=label, marker_color=c[name],
                             hovertemplate="%{x}: %{y:+.1%}<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, barcornerradius=4)
    fig.add_hline(y=0, line=dict(color=c["axis"], width=1))
    return _style(fig, c, "+.0%", height=300)


def risk_histogram(risk_pct: pd.Series, target: float, mode: str) -> go.Figure:
    """How much of account equity each entry put at risk, with the target marked."""
    c = colors(mode)
    fig = go.Figure(go.Histogram(x=risk_pct, marker_color=c["portfolio"],
                                 xbins=dict(start=0, size=0.0025),
                                 hovertemplate="%{x:.2%} risk: %{y} entries<extra></extra>"))
    fig.add_vline(x=target, line=dict(color=c["ink"], width=1.5, dash="dash"))
    fig.add_annotation(x=target, y=1, yref="paper", text=f"{target:.0%} target", showarrow=False,
                       xanchor="left", xshift=4, yanchor="top", font=dict(color=c["ink"], size=12))
    fig.update_layout(bargap=0.08, barcornerradius=4, showlegend=False)
    fig.update_xaxes(tickformat=".0%", title=dict(text="Risk at entry, % of account equity",
                                                  font=dict(color=c["muted"], size=12)))
    return _style(fig, c, "d", height=280)


def exposure_chart(table: pd.DataFrame, mode: str) -> go.Figure:
    """Horizontal bars of weights (largest at the top), labelled with the percentage."""
    c = colors(mode)
    table = table.sort_values("weight")
    fig = go.Figure(go.Bar(
        x=table["weight"], y=[str(i) for i in table.index], orientation="h",
        marker_color=c["portfolio"], text=[f"{w:.0%}" for w in table["weight"]],
        textposition="outside", textfont=dict(color=c["ink"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    _style(fig, c, "", height=max(160, 44 * len(table)))
    fig.update_layout(barcornerradius=4, showlegend=False, margin=dict(l=8, r=40, t=8, b=8))
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, table["weight"].max() * 1.2])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
