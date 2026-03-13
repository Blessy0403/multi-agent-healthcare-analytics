"""Thesis-level chart utilities: publication-quality Plotly figures."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

# Thesis-style layout: clean, readable, publication-ready
THESIS_LAYOUT = dict(
    font=dict(family="Georgia, 'Times New Roman', serif", size=13),
    title_font=dict(size=18, color="#1a1a1a"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=70, r=40, t=60, b=60),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.2)",
        title_font=dict(size=14),
        tickfont=dict(size=12),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.2)",
        title_font=dict(size=14),
        tickfont=dict(size=12),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
    ),
    hovermode="x unified",
    showlegend=True,
)

# Academic color palette (colorblind-friendly, print-safe)
COLORS = {
    "primary": "#2166ac",
    "secondary": "#4393c3",
    "accent": "#92c5de",
    "neutral": "#666666",
    "success": "#1b7837",
    "warning": "#d95f02",
    "series": ["#2166ac", "#4393c3", "#92c5de", "#d6604d", "#f4a582"],
}

# Distinct color per training model (thesis-grade 6-model suite)
MODEL_COLORS = {
    "logistic_regression": "#2166ac",   # Blue
    "random_forest": "#1b7837",        # Green
    "xgboost": "#d95f02",              # Orange
    "svm": "#7570b3",                  # Purple
    "gradient_boosting": "#e7298a",     # Magenta
    "knn": "#66a61e",                  # Olive
}

# Agent colors for 4-agent pipeline (thesis: Data → Model → Explainability → Evaluation)
AGENT_COLORS = {
    "data_agent": "#3182ce",
    "model_agent": "#238b45",
    "explainability_agent": "#6a51a3",
    "evaluation_agent": "#b45309",
}


def _model_color(model_name: str) -> str:
    """Return color for a model (default primary if unknown)."""
    key = (model_name or "").lower().replace(" ", "_")
    return MODEL_COLORS.get(key, COLORS["primary"])


def _agent_color(agent_name: str) -> str:
    """Return color for an agent."""
    key = (agent_name or "").lower().replace(" ", "_")
    return AGENT_COLORS.get(key, COLORS["neutral"])


def _hex_lighten(hex_color: str, factor: float) -> str:
    """Lighten a hex color by factor (0–1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def apply_thesis_theme(fig: go.Figure, title: str = None, height: int = 480, width: int = 800) -> go.Figure:
    """Apply publication-quality layout to a Plotly figure (plotly_white template, consistent margins)."""
    fig.update_layout(template="plotly_white", **THESIS_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    fig.update_layout(height=height, width=width)
    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str = ""
) -> go.Figure:
    """ROC curve with model-specific color so each model looks distinct."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    line_color = _model_color(model_name)
    display_name = (model_name or "").replace("_", " ").title()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{display_name} (AUC = {roc_auc:.3f})",
            line=dict(color=line_color, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        )
    )
    fig.update_layout(
        title=dict(text=f"ROC Curve — {display_name}", x=0.5, xanchor="center"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return apply_thesis_theme(fig, height=500)


def plot_pr_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str = ""
) -> go.Figure:
    """Precision-Recall curve; useful for imbalanced data (positive class = minority)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    line_color = _model_color(model_name)
    display_name = (model_name or "").replace("_", " ").title()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"{display_name} (AP = {ap:.3f})",
            line=dict(color=line_color, width=2.5),
        )
    )
    fig.update_layout(
        title=dict(text=f"Precision-Recall Curve — {display_name}", x=0.5, xanchor="center"),
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return apply_thesis_theme(fig, height=500)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str] = None, model_name: str = ""
) -> go.Figure:
    """Confusion matrix heatmap with model-specific colorscale."""
    if class_names is None:
        class_names = ["Negative", "Positive"]
    # Colorscale from white to model color
    base = _model_color(model_name)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale=[[0, "#f7fafc"], [0.5, _hex_lighten(base, 0.6)], [1, base]],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=16),
            showscale=True,
            colorbar=dict(title="Count", len=0.6),
        )
    )
    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    return apply_thesis_theme(fig, height=420)


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance",
    model_name: str = "",
) -> go.Figure:
    """Horizontal bar chart for feature importance with model-specific color."""
    sorted_idx = np.argsort(importance_values)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = importance_values[sorted_idx]
    base = _model_color(model_name)
    light = _hex_lighten(base, 0.7)

    fig = go.Figure(
        data=go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation="h",
            marker=dict(
                color=sorted_values,
                colorscale=[[0, light], [0.5, base], [1, base]],
                line=dict(width=0),
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
    )
    return apply_thesis_theme(fig, height=max(400, len(feature_names) * 28))


def plot_agent_timeline(agent_times: Dict[str, float]) -> go.Figure:
    """Agent execution times bar chart — each agent gets its own color (multimodal pipeline)."""
    agents = list(agent_times.keys())
    times = list(agent_times.values())
    agent_labels = [a.replace("_", " ").title() for a in agents]
    colors = [_agent_color(a) for a in agents]

    fig = go.Figure(
        data=go.Bar(
            x=agent_labels,
            y=times,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{t:.2f}s" for t in times],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Agent Execution Times", x=0.5, xanchor="center"),
        xaxis_title="Agent",
        yaxis_title="Time (s)",
        showlegend=False,
    )
    return apply_thesis_theme(fig, height=400)


# Colors for comparison chart: Multi Agent vs Baseline vs Difference (distinct series)
COMPARISON_SERIES_COLORS = {
    "multi_agent": "#2166ac",   # Blue
    "baseline": "#737373",      # Neutral gray
    "difference": "#9970ab",    # Purple
}


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: distinct colors per series (Multi Agent, Baseline, Difference) and per model when applicable."""
    fig = go.Figure()
    model_col = "model" if "model" in metrics_df.columns else metrics_df.columns[0]
    models = metrics_df[model_col].astype(str).tolist()
    numeric_cols = [c for c in metrics_df.columns if c != model_col and pd.api.types.is_numeric_dtype(metrics_df[c])]
    if not numeric_cols:
        numeric_cols = [c for c in metrics_df.columns if c != model_col]
    model_colors = [_model_color(m) for m in models] if models else []
    for col in numeric_cols:
        # Use distinct color per comparison series (multi_agent, baseline, difference); else per-model colors
        col_lower = col.lower().replace(" ", "_")
        if col_lower in COMPARISON_SERIES_COLORS:
            bar_color = [COMPARISON_SERIES_COLORS[col_lower]] * len(models)
        else:
            bar_color = model_colors if len(model_colors) == len(models) else COLORS["primary"]
        fig.add_trace(
            go.Bar(
                name=col.replace("_", " ").title(),
                x=models,
                y=metrics_df[col].tolist(),
                marker_color=bar_color,
            )
        )
    fig.update_layout(
        title=dict(text="Model Metrics Comparison", x=0.5, xanchor="center"),
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
    )
    return apply_thesis_theme(fig, height=480)


def plot_distribution_histogram(
    series: pd.Series, xlabel: str, title: str, color: str = None
) -> go.Figure:
    """Thesis-style histogram for feature/target distributions."""
    color = color or COLORS["primary"]
    fig = go.Figure(
        data=go.Histogram(
            x=series,
            nbinsx=min(35, max(15, int(np.sqrt(len(series))))),
            marker=dict(color=color, line=dict(color="white", width=0.5)),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=xlabel,
        yaxis_title="Count",
        showlegend=False,
    )
    return apply_thesis_theme(fig, height=400)


def plot_class_pie(labels: List[Any], values: List[float], title: str) -> go.Figure:
    """Pie chart for class distribution with thesis styling."""
    fig = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=COLORS["series"], line=dict(color="white", width=2)),
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return apply_thesis_theme(fig, height=420)


def plot_missing_values_bar(missing_df: pd.DataFrame) -> go.Figure:
    """Bar chart for missing values per feature."""
    fig = go.Figure(
        data=go.Bar(
            x=missing_df.iloc[:, 0],
            y=missing_df.iloc[:, 1],
            marker=dict(color=COLORS["warning"], line=dict(width=0)),
        )
    )
    fig.update_layout(
        title=dict(text="Missing Values by Feature", x=0.5, xanchor="center"),
        xaxis_title="Feature",
        yaxis_title="Missing count",
    )
    return apply_thesis_theme(fig, height=400)
