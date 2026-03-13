"""Reusable UI components for thesis-worthy, consistent dashboard styling."""

import streamlit as st
from typing import Optional


def status_chip(label: str, value: str, status: str = "info") -> None:
    """Render a compact status chip. status in {'info', 'success', 'warning'}."""
    colors = {
        "info": ("#3182ce", "#ebf8ff"),
        "success": ("#38a169", "#f0fff4"),
        "warning": ("#d69e2e", "#fffff0"),
    }
    border, bg = colors.get(status, colors["info"])
    st.markdown(
        f'<div style="'
        f'display:inline-block; padding:0.25rem 0.6rem; border-radius:999px; '
        f'background:{bg}; border:1px solid {border}; font-size:0.8rem; '
        f'color:#2d3748; margin-bottom:0.35rem;">'
        f'<span style="color:#718096;">{label}:</span> <strong>{value}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str | float | int, subtitle: Optional[str] = None, max_value_len: int = 18) -> None:
    """Render a KPI card with consistent height and optional truncation for long values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
                value = str(int(value))
            else:
                value = f"{float(value):.3f}"
        except (TypeError, ValueError):
            value = str(value)
    else:
        value = str(value)
    if len(value) > max_value_len:
        display_value = value[: max_value_len - 2] + "…"
        title_attr = f' title="{value}"'
    else:
        display_value = value
        title_attr = ""
    sub = f'<p style="margin:0.25rem 0 0 0; font-size:0.75rem; color:#718096;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="kpi-card" style="'
        f'background:white; border-radius:12px; padding:1rem 1.25rem; '
        f'border:1px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.06); '
        f'min-height:5.5rem; display:flex; flex-direction:column; justify-content:center;">'
        f'<div style="font-size:0.8rem; color:#718096; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">{title}</div>'
        f'<div style="font-size:1.5rem; font-weight:700; color:#1a365d;"{title_attr}>{display_value}</div>'
        f'{sub}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: Optional[str] = None) -> None:
    """Render a section header with optional caption."""
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def render_context_bar(
    run_label: str,
    dataset: str,
    best_model: Optional[str] = None,
    feedback_mode: Optional[str] = None,
    retraining_performed: Optional[str] = None,
) -> None:
    """Render a compact context bar with pill badges: Run | Dataset | Feedback | Retraining | Best model."""
    pills = [
        ("Run", run_label[:24] + ("…" if len(run_label) > 24 else "")),
        ("Dataset", dataset.replace("_", " ").title() if isinstance(dataset, str) else str(dataset)),
    ]
    if feedback_mode is not None:
        pills.append(("Feedback", feedback_mode))
    if retraining_performed is not None:
        pills.append(("Retraining", retraining_performed))
    if best_model:
        best_label = "Best model (after feedback)" if (feedback_mode and feedback_mode != "—") else "Best model"
        pills.append((best_label, best_model.replace("_", " ").title()))
    html_pills = " ".join(
        f'<span style="background:#e2e8f0; color:#4a5568; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">{k}: {v}</span>'
        for k, v in pills
    )
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem; margin-bottom:0.75rem;">{html_pills}</div>',
        unsafe_allow_html=True,
    )


def viewing_model_badge(selected_model: str, best_model: Optional[str]) -> None:
    """If selected_model != best_model, show subtle badge 'Viewing: X (not best)'."""
    if not best_model or selected_model == best_model:
        return
    sel = selected_model.replace("_", " ").title()
    st.markdown(
        f'<div style="font-size:0.85rem; color:#718096;">'
        f'Viewing: <strong>{sel}</strong> <span style="color:#a0aec0;">(not best)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
