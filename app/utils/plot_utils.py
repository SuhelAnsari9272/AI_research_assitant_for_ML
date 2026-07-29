"""Reusable Plotly figure builders. No page/component should build a raw
plotly figure inline - route it through here so styling stays consistent.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#4F46E5"
SECONDARY = "#0EA5E9"
MUTED = "#94A3B8"
DANGER = "#DC2626"

_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#1E293B"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=320,
)


def _apply_defaults(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def numeric_distribution(series: pd.Series, title: str = "Distribution") -> go.Figure:
    """Histogram + rug for a numeric feature."""
    fig = px.histogram(series.dropna(), nbins=40, color_discrete_sequence=[PRIMARY])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Count")
    return _apply_defaults(fig, title)


def categorical_distribution(series: pd.Series, title: str = "Top Categories", top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of the most frequent categories."""
    counts = series.value_counts(dropna=True).head(top_n).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=[str(v) for v in counts.index],
            orientation="h",
            marker_color=SECONDARY,
        )
    )
    fig.update_layout(xaxis_title="Count", yaxis_title=None)
    return _apply_defaults(fig, title)


def datetime_distribution(series: pd.Series, title: str = "Records over time") -> go.Figure:
    """Line chart of record counts binned by date."""
    dt = pd.to_datetime(series, errors="coerce").dropna()
    counts = dt.dt.to_period("M").value_counts().sort_index()
    fig = go.Figure(
        go.Scatter(
            x=counts.index.astype(str),
            y=counts.values,
            mode="lines+markers",
            line=dict(color=PRIMARY, width=2),
        )
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Count")
    return _apply_defaults(fig, title)


def missingness_bar(missing_pct_by_col: pd.Series, title: str = "Missing % by column") -> go.Figure:
    data = missing_pct_by_col[missing_pct_by_col > 0].sort_values(ascending=True)
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected", showarrow=False, font=dict(size=13))
        return _apply_defaults(fig, title)
    fig = go.Figure(go.Bar(x=data.values, y=data.index, orientation="h", marker_color=DANGER))
    fig.update_layout(xaxis_title="% missing", yaxis_title=None)
    return _apply_defaults(fig, title)


def confusion_matrix_heatmap(matrix, labels, title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            colorscale="Blues",
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return _apply_defaults(fig, title)


def residuals_scatter(y_true, y_pred, title: str = "Predicted vs Actual") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(color=PRIMARY, opacity=0.6)))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dash")))
    fig.update_layout(showlegend=False, xaxis_title="Actual", yaxis_title="Predicted")
    return _apply_defaults(fig, title)


def model_comparison_bar(names, scores, metric_name: str, title: str = "Model Comparison") -> go.Figure:
    fig = go.Figure(go.Bar(x=names, y=scores, marker_color=[PRIMARY if i == 0 else MUTED for i in range(len(names))]))
    fig.update_layout(yaxis_title=metric_name, xaxis_title=None)
    return _apply_defaults(fig, title)
