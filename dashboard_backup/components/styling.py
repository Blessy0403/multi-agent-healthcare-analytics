"""Thesis-level dashboard styling: clean, professional, aesthetic."""

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS for an aesthetic, thesis-appropriate dashboard."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean, readable base */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 2rem 2.5rem 3rem;
        background: #fafbfc;
        min-height: 100vh;
    }
    
    /* Page titles */
    h1 {
        color: #1a1a2e;
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #2d3748;
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Metric cards and KPI cards – same height, aligned */
    [data-testid="stMetricContainer"],
    .kpi-card {
        min-height: 5.5rem;
    }
    [data-testid="stMetricContainer"] {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a365d;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stSelectbox label { font-weight: 600; color: #2d3748; }
    
    /* Alerts */
    .stSuccess { border-radius: 10px; border-left: 4px solid #38a169; }
    .stInfo { border-radius: 10px; border-left: 4px solid #3182ce; }
    .stWarning { border-radius: 10px; border-left: 4px solid #d69e2e; }
    .stError { border-radius: 10px; border-left: 4px solid #e53e3e; }
    
    /* DataFrames */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .dataframe thead tr th { background: #2d3748; color: white; font-weight: 600; padding: 0.75rem; }
    .dataframe tbody tr:hover { background: #f7fafc; }
    
    /* Dividers */
    hr { border: none; height: 1px; background: #e2e8f0; margin: 1.5rem 0; }
    
    /* Expanders */
    .streamlit-expanderHeader { font-weight: 600; color: #2d3748; }
    
    /* Code */
    code { background: #edf2f7; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; color: #2d3748; }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """Render a clean header section."""
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <h1 style="margin: 0;">{title}</h1>
        {f'<p style="margin: 0.4rem 0 0 0; color: #718096; font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# Agent accent colors (Data → Model → Explainability → Evaluation → Feedback → CrossDataset)
AGENT_ACCENTS = {
    "data_agent": "#3182ce",
    "model_agent": "#238b45",
    "explainability_agent": "#6a51a3",
    "evaluation_agent": "#b45309",
    "feedback_agent": "#c53030",
    "cross_dataset_agent": "#2c5282",
}


def render_pipeline_flow():
    """Render pipeline flow: Data → Model → Explainability → Evaluation → Feedback (if enabled) → CrossDataset (if enabled)."""
    enable_feedback = True
    enable_cross_dataset = False
    run_id = st.session_state.get("run_id")
    if run_id:
        try:
            from dashboard.components.layout import get_run_metadata
            meta = get_run_metadata(run_id)
            if meta:
                enable_feedback = meta.get("enable_feedback", True)
                enable_cross_dataset = meta.get("enable_cross_dataset", False)
        except Exception:
            pass
    pills = [
        '<span class="agent-pill" style="background:#3182ce;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">📊 Data Agent</span>',
        '<span class="agent-pill" style="background:#238b45;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">🤖 Model Agent</span>',
        '<span class="agent-pill" style="background:#6a51a3;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">🔍 Explainability Agent</span>',
        '<span class="agent-pill" style="background:#b45309;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">⚖️ Evaluation Agent</span>',
    ]
    if enable_feedback:
        pills.append('<span class="agent-pill" style="background:#c53030;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">🔄 Feedback Agent</span>')
    if enable_cross_dataset:
        pills.append('<span class="agent-pill" style="background:#2c5282;color:white;padding:0.5rem 1rem;border-radius:20px;font-weight:600;font-size:0.9rem;">📂 Cross-Dataset Agent</span>')
    arrow = '<span style="color:#94a3b8;font-size:1.2rem;">→</span>'
    inner = arrow.join(pills)
    st.markdown(f"""
    <div class="pipeline-flow" style="
        display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        flex-wrap: wrap; margin: 1rem 0 1.5rem 0; padding: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px; border: 1px solid #e2e8f0;
    ">{inner}</div>
    """, unsafe_allow_html=True)


def render_agent_section(agent_name: str, accent_hex: str):
    """Render a section header styled for an agent (full pipeline: Data, Model, Explainability, Evaluation, Feedback, CrossDataset)."""
    labels = {
        "data_agent": "📊 Data Agent",
        "model_agent": "🤖 Model Agent",
        "explainability_agent": "🔍 Explainability Agent",
        "evaluation_agent": "⚖️ Evaluation Agent",
        "feedback_agent": "🔄 Feedback Agent",
        "cross_dataset_agent": "📂 Cross-Dataset Agent",
    }
    label = labels.get(agent_name.lower().replace(" ", "_"), agent_name)
    st.markdown(f"""
    <div style="
        border-left: 4px solid {accent_hex};
        background: linear-gradient(90deg, {accent_hex}08 0%, transparent 100%);
        padding: 0.75rem 1rem; margin: 1.25rem 0 0.75rem 0;
        border-radius: 0 8px 8px 0; font-weight: 600; color: #1a1a2e;
    ">{label}</div>
    """, unsafe_allow_html=True)
