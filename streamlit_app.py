"""
Athlete Wellness & Mood Analytics — Streamlit Dashboard
=========================================================
Run with:  streamlit run streamlit_app.py
Requires:  pip install streamlit plotly pandas numpy scikit-learn
All pipeline modules (task1–task4_7) must be in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

sys.path.insert(0, os.path.dirname(__file__))

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Athlete Wellness Analytics",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0f172a; color: #e2e8f0; }
  [data-testid="stSidebar"] { background-color: #1e293b; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .kpi-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 14px; padding: 18px 14px;
    text-align: center; margin-bottom: 8px;
  }
  .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.1; }
  .kpi-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;
               letter-spacing: 0.06em; margin-top: 4px; }
  .section-header {
    font-size: 1rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .06em;
    border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 12px;
  }
  .alert-high { background:#7f1d1d33; border-left:4px solid #ef4444;
                border-radius:8px; padding:10px 14px; margin:4px 0; }
  .alert-mod  { background:#78350f33; border-left:4px solid #f59e0b;
                border-radius:8px; padding:10px 14px; margin:4px 0; }
  .alert-ok   { background:#14532d33; border-left:4px solid #22c55e;
                border-radius:8px; padding:10px 14px; margin:4px 0; }
  h1,h2,h3 { color: #f1f5f9 !important; }
  hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
CACHE_FILE = os.path.join(os.path.dirname(__file__), "analytics_pipeline_output.csv")
RAW_CACHE  = os.path.join(os.path.dirname(__file__), "raw_wearable_data.csv")

@st.cache_data(show_spinner="Running analytics pipeline…")
def load_pipeline_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    from pipeline import run_full_pipeline
    df = run_full_pipeline()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(show_spinner="Loading raw sensor data…")
def load_raw_data():
    if os.path.exists(RAW_CACHE):
        df = pd.read_csv(RAW_CACHE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None

df  = load_pipeline_data()
raw = load_raw_data()

@st.cache_data
def generate_pdf_report(pipeline_df, raw_df):
    if pipeline_df.empty:
        return None

    pipeline = pipeline_df.copy()
    raw = raw_df.copy() if raw_df is not None else None

    period = f"{pipeline['date'].min().strftime('%b %d, %Y')} - {pipeline['date'].max().strftime('%b %d, %Y')}"

    # Key summary values
    avg_readiness = pipeline['readiness_score'].mean()
    high_readiness_days = int((pipeline['readiness_label'] == 'high').sum())
    optimal_recovery_days = int((pipeline['recovery_state'] == 'optimal').sum())
    injury_events = int(pipeline['injury_event'].sum())
    high_injury_risk_days = int((pipeline['injury_risk_label'] == 'high').sum())
    acwr_avg = pipeline['acwr'].mean()
    acwr_danger_days = int((pipeline['acwr'] > 1.5).sum())

    insights = []
    insights.append(f"The athlete's readiness has an average score of {avg_readiness:.1f}/100 over the period {period}.")
    if avg_readiness >= 75:
        insights.append('Overall readiness is strong; training load is being managed well.')
    elif avg_readiness >= 50:
        insights.append('Readiness is moderate; monitor recovery and fatigue markers closely.')
    else:
        insights.append('Readiness is low; implement recovery and load reduction strategies.')

    insights.append(f"There were {high_readiness_days} high-readiness days and {optimal_recovery_days} optimal recovery days.")
    insights.append(f"Injury events: {injury_events} and high-injury-risk days: {high_injury_risk_days}.")
    insights.append(f"Mean ACWR is {acwr_avg:.2f}, with {acwr_danger_days} day(s) in overreaching zone (>1.5).")

    plt.style.use('classic')

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.93, 'Athlete Wellness & Mood Analytics – Executive Report', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(0.5, 0.87, f'Date range: {period}', ha='center', va='center', fontsize=12)

        bullet_y = 0.78
        for line in [
            f'Total days analyzed: {len(pipeline)}',
            f'Average readiness score: {avg_readiness:.1f}/100',
            f'High readiness days: {high_readiness_days}',
            f'Optimal recovery days: {optimal_recovery_days}',
            f'Injury events: {injury_events}',
            f'High injury risk days: {high_injury_risk_days}',
            f'Average ACWR: {acwr_avg:.2f}',
        ]:
            ax.text(0.09, bullet_y, f'• {line}', fontsize=12)
            bullet_y -= 0.04

        ax.text(0.09, bullet_y - 0.02, 'Key interpretations', fontsize=14, fontweight='bold')
        bullet_y -= 0.07
        for line in insights:
            ax.text(0.09, bullet_y, f'• {line}', fontsize=12)
            bullet_y -= 0.04

        ax.text(0.09, 0.05, 'Generated from MoodChanger-main pipeline and raw wearable data.', fontsize=10, color='gray')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.plot(pipeline['date'], pipeline['readiness_score'], color='#1f77b4', lw=2, label='Daily Readiness')
        ax.plot(pipeline['date'], pipeline['readiness_score'].rolling(7, min_periods=1).mean(), color='#ff7f0e', lw=1.5, linestyle='--', label='7-day average')
        ax.axhline(75, color='green', linestyle='--', label='High threshold')
        ax.axhline(50, color='orange', linestyle='--', label='Moderate threshold')
        ax.set_title('Readiness Score Trend', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Readiness Score')
        ax.set_ylim(0, 110)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, 'Interpretation: Look for downward patterns before injury spikes; aim to keep score >70', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
        ax1.plot(pipeline['date'], pipeline['acute_load'], label='Acute (7d)', color='#ff7f0e', lw=2)
        ax1.plot(pipeline['date'], pipeline['chronic_load'], label='Chronic (28d)', color='#2ca02c', lw=2)
        ax1.set_title('Training Load: Acute vs Chronic', fontsize=16)
        ax1.set_ylabel('Load (AU)')
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        ax2.bar(pipeline['date'], pipeline['acwr'], color=['red' if v > 1.5 else 'orange' if v > 1.3 else 'green' if v >= 0.8 else 'blue' for v in pipeline['acwr']])
        ax2.axhline(1.3, color='green', linestyle='--', label='Optimal upper')
        ax2.axhline(1.5, color='red', linestyle='--', label='Overreaching')
        ax2.axhline(0.8, color='blue', linestyle='--', label='Undertraining')
        ax2.set_title('ACWR (Acute:Chronic Workload Ratio)', fontsize=16)
        ax2.set_ylabel('ACWR')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.25)
        ax2.text(0.02, 0.90, 'Interpretation: Sustained 0.8-1.3 is optimal; >1.5 high risk', transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
        ax1.bar(pipeline['date'], pipeline['injury_risk_score'], color='#d62728', alpha=0.7)
        ax1.set_title('Injury Risk Score Timeline', fontsize=16)
        ax1.set_ylabel('Risk Score')
        ax1.axhline(60, color='black', linestyle='--', label='High risk threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        scatter = ax2.scatter(pipeline['acwr'], pipeline['soreness_score'], c=pipeline['injury_risk_score'], cmap='coolwarm', s=70, edgecolors='k', alpha=0.85)
        ax2.set_title('ACWR vs Soreness vs Injury Risk', fontsize=16)
        ax2.set_xlabel('ACWR')
        ax2.set_ylabel('Soreness (1-10)')
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label('Injury Risk Score')
        ax2.grid(True, alpha=0.25)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if raw is not None and not raw.empty:
            sample_day = raw['timestamp'].dt.date.max()
            sample = raw[raw['timestamp'].dt.date == sample_day]
            if not sample.empty:
                fig, axs = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
                axs[0].plot(sample['timestamp'], sample['HR_bpm'], color='#ff7f0e')
                axs[0].set_title(f'HR (bpm) for {sample_day}')
                axs[0].set_ylabel('HR')
                axs[1].plot(sample['timestamp'], sample['HRV_ms'], color='#1f77b4')
                axs[1].set_title('HRV (ms)')
                axs[1].set_ylabel('HRV')
                axs[2].plot(sample['timestamp'], sample['accel_g'], color='#2ca02c', alpha=0.8)
                axs[2].set_title('Acceleration (g)')
                axs[2].set_ylabel('Accel')
                axs[3].plot(sample['timestamp'], sample['skin_temp_C'], color='#d62728')
                axs[3].set_title('Skin temperature (°C)')
                axs[3].set_ylabel('Temp')
                axs[3].set_xlabel('Time')
                for ax in axs: ax.grid(True, alpha=0.2)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()

# ══════════════════════════════════════════════════════════════════
# COLOURS & LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════
C = {
    "high": "#22c55e", "moderate": "#f59e0b", "low": "#ef4444",
    "optimal": "#22c55e", "partial": "#a78bfa", "poor": "#ef4444",
    "accent": "#38bdf8", "accent2": "#818cf8",
    "orange": "#fb923c", "pink": "#f472b6",
    "bg": "#1e293b", "grid": "#334155",
}

PL = dict(
    paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#334155", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
)

def rc(s): return {"optimal":C["high"],"partial":C["partial"],"poor":C["low"]}.get(s,"#94a3b8")
def rlc(s): return C["high"] if s=="high" else C["moderate"] if s=="moderate" else C["low"]
def lc(v):
    if pd.isna(v): return "#94a3b8"
    return "#60a5fa" if v<0.8 else C["high"] if v<=1.3 else C["moderate"] if v<=1.5 else C["low"]

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  Wellness Analytics")
    st.markdown("---")
    st.markdown("**Patient Profile**")
    st.markdown("**ID:** SYN-2024-001")
    st.markdown("**Name:** Alex Mora")
    st.markdown("**Sport:** Triathlon · Age 28")
    st.markdown("---")
    st.markdown("**Date Range**")
    d_min = df["date"].min().date()
    d_max = df["date"].max().date()
    start_d, end_d = st.date_input("Select range", value=(d_min, d_max),
                                    min_value=d_min, max_value=d_max)
    st.markdown("---")
    st.markdown("**Display Options**")
    show_inj = st.toggle("Show injury markers", value=True)
    show_thr = st.toggle("Show threshold lines", value=True)
    smooth7  = st.toggle("7-day smoothing on readiness", value=False)
    st.markdown("---")
    st.markdown("**Day Category Filter**")
    all_cats = sorted(df["day_category"].unique().tolist())
    sel_cats = st.multiselect("Include", all_cats, default=all_cats)
    st.markdown("---")
    st.caption("Mood & Wellness Analytics Pipeline\nSynthetic data — demo only")

# Apply filters
fdf = df[
    (df["date"].dt.date >= start_d) &
    (df["date"].dt.date <= end_d) &
    (df["day_category"].isin(sel_cats))
].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("#  Athlete Wellness & Mood Analytics Dashboard")
st.markdown(
    f"**{len(fdf)} days** analysed &nbsp;·&nbsp; "
    f"{fdf['date'].min().strftime('%b %d')} – {fdf['date'].max().strftime('%b %d, %Y')} "
    f"&nbsp;·&nbsp; Patient SYN-2024-001 (Alex Mora)"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ALERTS + RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
if len(fdf):
    latest = fdf.iloc[-1]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="section-header"> Current Status Alerts</div>', unsafe_allow_html=True)
        rl = latest["readiness_label"]
        rc_s = latest["recovery_state"]
        ir = latest["injury_risk_label"]
        acwr_v = latest["acwr"]
        cls_map = {"high":"alert-ok","moderate":"alert-mod","low":"alert-high",
                   "optimal":"alert-ok","partial":"alert-mod","poor":"alert-high"}
        st.markdown(
            f'<div class="{cls_map.get(rl,"alert-ok")}"> <b>Readiness:</b> '
            f'{latest["readiness_score"]:.0f}/100 — <b>{rl.upper()}</b></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="{cls_map.get(rc_s,"alert-ok")}"> <b>Recovery:</b> <b>{rc_s.upper()}</b> '
            f'· Sleep: {latest["sleep_duration_h"]:.1f}h</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="{cls_map.get(ir,"alert-ok")}"> <b>Injury Risk:</b> '
            f'{latest["injury_risk_score"]:.0f}/100 — <b>{ir.upper()}</b> · ACWR: {acwr_v:.2f}</div>',
            unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="section-header"> Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if rl == "low":        recs.append(" Skip high-intensity training today")
        elif rl == "moderate": recs.append(" Moderate session — monitor HR zones")
        else:                  recs.append(" Ready for full training load")
        if rc_s == "poor":     recs.append(" Prioritise sleep (aim 7.5h+)")
        if float(acwr_v) > 1.4: recs.append(" Reduce training volume this week")
        if ir == "high":       recs.append(" High injury risk : rest recommended")
        if not recs:           recs.append(" All systems nominal")
        for r in recs:
            st.markdown(f"- {r}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"> Key Performance Indicators</div>', unsafe_allow_html=True)
cols = st.columns(6)
kpi_data = [
    (f"{fdf['readiness_score'].mean():.0f}", "Avg Readiness",
     rlc("high") if fdf["readiness_score"].mean()>=75 else rlc("moderate") if fdf["readiness_score"].mean()>=50 else rlc("low")),
    (str(int((fdf["readiness_label"]=="high").sum())),    "High Readiness Days",    C["high"]),
    (str(int((fdf["recovery_state"]=="optimal").sum())),  "Optimal Recovery Days",  C["optimal"]),
    (str(int((fdf["load_balance_label"]=="optimal").sum())), "Optimal Load Days",   C["accent"]),
    (str(int(fdf["injury_event"].sum())),                 "Injury Events",          C["low"]),
    (str(int((fdf["injury_risk_label"]=="high").sum())),  "High Risk Days",         C["moderate"]),
]
for col, (val, label, color) in zip(cols, kpi_data):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{val}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Readiness", " Sleep & Recovery",
    " Training Load", " Injury Risk", " Raw Signals"
])

# ─── TAB 1: READINESS ──────────────────────────────────────────
with tab1:
    st.markdown("### Readiness Score Trend")
    y = fdf["readiness_score"].rolling(7,min_periods=1).mean() if smooth7 else fdf["readiness_score"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=y,
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color=C["accent"],width=2.5), name="Readiness",
        hovertemplate="<b>%{x|%b %d}</b><br>Score: %{y:.1f}<extra></extra>"))
    for lbl, col in [("high",C["high"]),("moderate",C["moderate"]),("low",C["low"])]:
        m = fdf["readiness_label"]==lbl
        fig.add_trace(go.Scatter(x=fdf["date"][m], y=fdf["readiness_score"][m],
            mode="markers", marker=dict(color=col,size=7), name=lbl.capitalize()))
    if show_thr:
        fig.add_hline(y=75, line_dash="dash", line_color=C["high"],   opacity=0.5, annotation_text="High (75)")
        fig.add_hline(y=50, line_dash="dash", line_color=C["moderate"],opacity=0.5, annotation_text="Moderate (50)")
    if show_inj:
        for _, row in fdf[fdf["injury_event"]==1].iterrows():
            fig.add_vline(x=row["date"], line_color="#f43f5e", line_dash="dot", opacity=0.7)
            fig.add_annotation(x=row["date"], y=105, text="⚡", showarrow=False, font=dict(size=14))
    fig.update_layout(**PL, height=380, yaxis=dict(range=[0,112],title="Score (0–100)",gridcolor=C["grid"]),
                      title="Daily Readiness Score")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=fdf["date"], y=fdf["HRV_ms"],
            fill="tozeroy", fillcolor="rgba(129,140,248,0.08)",
            line=dict(color=C["accent2"],width=2), name="HRV (ms)"))
        f2.add_trace(go.Scatter(x=fdf["date"], y=fdf["HRV_ms"].rolling(7,min_periods=1).mean(),
            line=dict(color="white",width=1.5,dash="dot"), name="7d avg"))
        f2.update_layout(**PL, height=280, title="HRV Trend (ms)", yaxis_title="RMSSD (ms)")
        st.plotly_chart(f2, width='stretch')
    with c2:
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=fdf["date"], y=fdf["resting_HR"],
            fill="tozeroy", fillcolor="rgba(251,146,60,0.08)",
            line=dict(color=C["orange"],width=2), name="Resting HR"))
        if show_thr:
            f3.add_hline(y=52, line_dash="dash", line_color=C["high"], opacity=0.5, annotation_text="Baseline 52")
        f3.update_layout(**PL, height=280, title="Resting Heart Rate (bpm)", yaxis_title="bpm")
        st.plotly_chart(f3, width='stretch')

# ─── TAB 2: SLEEP & RECOVERY ───────────────────────────────────
with tab2:
    st.markdown("### Sleep Duration & Recovery State")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.05)
    sleep_cols = [C["high"] if h>=7.5 else C["moderate"] if h>=6 else C["low"] for h in fdf["sleep_duration_h"]]
    fig.add_trace(go.Bar(x=fdf["date"], y=fdf["sleep_duration_h"],
        marker_color=sleep_cols, name="Sleep (h)",
        hovertemplate="<b>%{x|%b %d}</b><br>Sleep: %{y:.1f}h<extra></extra>"), row=1, col=1)
    if show_thr:
        fig.add_hline(y=7.5, line_dash="dash", line_color=C["high"],  opacity=0.6, annotation_text="Target 7.5h", row=1, col=1)
        fig.add_hline(y=6.0, line_dash="dash", line_color=C["low"],   opacity=0.6, annotation_text="Min 6.0h",    row=1, col=1)
    fig.add_trace(go.Bar(x=fdf["date"], y=[1]*len(fdf),
        marker_color=[rc(s) for s in fdf["recovery_state"]], name="Recovery",
        customdata=fdf["recovery_state"],
        hovertemplate="<b>%{x|%b %d}</b><br>%{customdata}<extra></extra>"), row=2, col=1)
    fig.update_layout(**PL, height=420, title="Sleep & Recovery State")
    fig.update_yaxes(title_text="Hours", row=1, col=1, gridcolor=C["grid"])
    fig.update_yaxes(visible=False, row=2, col=1)
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        f4 = px.scatter(fdf, x="sleep_duration_h", y="readiness_score", color="recovery_state",
            color_discrete_map={"optimal":C["high"],"partial":C["partial"],"poor":C["low"]},
            title="Sleep Duration vs Readiness", labels={"sleep_duration_h":"Sleep (h)","readiness_score":"Readiness"}, template="plotly_dark")
        f4.update_layout(**PL, height=300)
        st.plotly_chart(f4, width='stretch')
    with c2:
        rc_counts = fdf["recovery_state"].value_counts()
        f5 = go.Figure(go.Pie(labels=rc_counts.index, values=rc_counts.values, hole=0.55,
            marker=dict(colors=[rc(s) for s in rc_counts.index]), textinfo="label+percent"))
        f5.update_layout(**PL, height=300, title="Recovery Distribution")
        st.plotly_chart(f5, width='stretch')

# ─── TAB 3: TRAINING LOAD ──────────────────────────────────────
with tab3:
    st.markdown("### Acute:Chronic Workload Ratio (ACWR)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55,0.45], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["acute_load"],
        name="Acute (7d)", line=dict(color=C["orange"],width=2.5),
        hovertemplate="Acute: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["chronic_load"],
        name="Chronic (28d)", line=dict(color=C["accent"],width=2.5),
        hovertemplate="Chronic: %{y:.2f}<extra></extra>"), row=1, col=1)
    acwr_vals = fdf["acwr"].fillna(1)
    fig.add_trace(go.Bar(x=fdf["date"], y=acwr_vals,
        marker_color=[lc(v) for v in acwr_vals], name="ACWR",
        customdata=fdf["load_balance_label"],
        hovertemplate="<b>%{x|%b %d}</b><br>ACWR: %{y:.2f}<br>%{customdata}<extra></extra>"), row=2, col=1)
    if show_thr:
        for y_val, col, name in [(0.8,"#60a5fa","<0.8"),(1.3,C["high"],"1.3"),(1.5,C["moderate"],"1.5")]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=col, opacity=0.5,
                          annotation_text=name, row=2, col=1)
    fig.update_layout(**PL, height=440, title="Training Load Balance")
    fig.update_yaxes(title_text="Load (AU)", row=1, col=1, gridcolor=C["grid"])
    fig.update_yaxes(title_text="ACWR", row=2, col=1, gridcolor=C["grid"])
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        lb_counts = fdf["load_balance_label"].value_counts()
        f6 = go.Figure(go.Pie(labels=lb_counts.index, values=lb_counts.values, hole=0.5,
            marker=dict(colors=[lc({"undertraining":0.5,"optimal":1.0,"high":1.4,"overreaching":1.8}.get(l,1.0)) for l in lb_counts.index]),
            textinfo="label+percent"))
        f6.update_layout(**PL, height=300, title="Load Zone Distribution")
        st.plotly_chart(f6, width='stretch')
    with c2:
        st.markdown("**ACWR Reference Zones**")
        st.dataframe(pd.DataFrame({
            "Range": ["< 0.8","0.8–1.3","1.3–1.5","> 1.5"],
            "Zone":  ["Undertraining","Optimal","High","Overreaching"],
            "Risk":  ["Fitness loss","Safe","Elevated","Danger zone"],
        }), width='stretch', hide_index=True)
        st.markdown("**Load Stats**")
        st.dataframe(fdf[["training_load_AU","acute_load","chronic_load","acwr"]].describe().round(2),
                     width='stretch')

# ─── TAB 4: INJURY RISK ────────────────────────────────────────
with tab4:
    st.markdown("### Injury Risk Score Timeline")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fdf["date"], y=fdf["injury_risk_score"].fillna(0),
        marker_color=[{"low":C["high"],"moderate":C["moderate"],"high":C["low"]}.get(r,C["high"]) for r in fdf["injury_risk_label"]],
        name="Injury Risk", hovertemplate="<b>%{x|%b %d}</b><br>Risk: %{y:.0f}/100<extra></extra>"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["injury_risk_score"].rolling(7,min_periods=1).mean(),
        line=dict(color="white",width=1.5,dash="dot"), name="7d trend"))
    if show_thr:
        fig.add_hline(y=60, line_dash="dash", line_color=C["low"],      opacity=0.7, annotation_text="High (60)")
        fig.add_hline(y=30, line_dash="dash", line_color=C["moderate"], opacity=0.7, annotation_text="Moderate (30)")
    if show_inj:
        for _, row in fdf[fdf["injury_event"]==1].iterrows():
            fig.add_annotation(x=row["date"], y=fdf["injury_risk_score"].max()+8,
                text="⚡ Injury", showarrow=True, arrowcolor="#f43f5e",
                font=dict(color="#f43f5e",size=11), arrowhead=2)
    fig.update_layout(**PL, height=380, yaxis=dict(range=[0,115],title="Risk Score",gridcolor=C["grid"]),
                      title="Injury Risk Score Timeline")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        f7 = px.scatter(fdf, x="acwr", y="soreness_score", color="injury_risk_label",
            color_discrete_map={"low":C["high"],"moderate":C["moderate"],"high":C["low"]},
            size="injury_risk_score", size_max=18, title="ACWR vs Soreness",
            labels={"acwr":"ACWR","soreness_score":"Soreness (1–10)"}, template="plotly_dark")
        f7.update_layout(**PL, height=300)
        st.plotly_chart(f7, width='stretch')
    with c2:
        f8 = px.scatter(fdf, x="readiness_score", y="injury_risk_score", color="injury_risk_label",
            color_discrete_map={"low":C["high"],"moderate":C["moderate"],"high":C["low"]},
            title="Readiness vs Injury Risk", template="plotly_dark")
        f8.update_layout(**PL, height=300)
        st.plotly_chart(f8, width='stretch')

    st.markdown("### ⚡ Injury Event Log")
    inj_df = fdf[fdf["injury_event"]==1][["date","day_category","readiness_score",
        "recovery_state","acwr","soreness_score","injury_risk_score"]].copy()
    inj_df["date"] = inj_df["date"].dt.strftime("%b %d, %Y")
    if len(inj_df):
        st.dataframe(inj_df, width='stretch', hide_index=True)
    else:
        st.success("No injury events in selected date range.")

# ─── TAB 5: RAW SIGNALS ────────────────────────────────────────
with tab5:
    st.markdown("### Raw Wearable Sensor Signals")
    if raw is not None:
        available_dates = sorted(raw["timestamp"].dt.date.unique())
        sel_date = st.selectbox("Select a day to inspect", options=available_dates,
            index=min(2, len(available_dates)-1),
            format_func=lambda d: pd.Timestamp(d).strftime("%A, %b %d %Y"))
        day_raw = raw[raw["timestamp"].dt.date == sel_date]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.38,0.32,0.30], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=day_raw["timestamp"], y=day_raw["HR_bpm"],
            line=dict(color=C["orange"],width=1.5), name="HR (bpm)",
            hovertemplate="%{x|%H:%M} — %{y:.0f} bpm<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=day_raw["timestamp"], y=day_raw["HRV_ms"],
            line=dict(color=C["accent2"],width=1.5), name="HRV (ms)",
            hovertemplate="%{x|%H:%M} — %{y:.1f} ms<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=day_raw["timestamp"], y=day_raw["accel_g"],
            line=dict(color=C["high"],width=1), fill="tozeroy", fillcolor="rgba(34,197,94,0.07)",
            name="Accel (g)"), row=3, col=1)
        fig.add_trace(go.Scatter(x=day_raw["timestamp"], y=day_raw["skin_temp_C"],
            line=dict(color=C["pink"],width=1.5), name="Temp (°C)"), row=3, col=1)
        fig.update_layout(**PL, height=480,
                          title=f"Sensor Signals — {pd.Timestamp(sel_date).strftime('%A, %b %d %Y')}")
        fig.update_yaxes(title_text="HR",   row=1, col=1, gridcolor=C["grid"])
        fig.update_yaxes(title_text="HRV",  row=2, col=1, gridcolor=C["grid"])
        fig.update_yaxes(title_text="Accel/Temp", row=3, col=1, gridcolor=C["grid"])
        st.plotly_chart(fig, width='stretch')

        col1, col2 = st.columns([2,1])
        with col1:
            ph = day_raw["phase"].value_counts().reset_index()
            f9 = px.bar(ph, x="phase", y="count", color="phase", template="plotly_dark",
                        title="Phase Distribution (readings count)")
            f9.update_layout(**PL, height=260, showlegend=False)
            st.plotly_chart(f9, width='stretch')
        with col2:
            st.markdown("**Daily Signal Stats**")
            st.dataframe(day_raw[["HR_bpm","HRV_ms","accel_g","skin_temp_C","SpO2_pct"]]
                .describe().round(2).loc[["mean","min","max","std"]], width='stretch')
    else:
        st.warning("Raw sensor data not found. Place raw_wearable_data.csv in the same folder.")

# ══════════════════════════════════════════════════════════════════
# FULL DATA TABLE
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander(" Full Daily Analytics Table"):
    disp = fdf[["date","day_category","resting_HR","HRV_ms","sleep_duration_h",
                "training_load_AU","readiness_score","readiness_label",
                "recovery_state","acwr","load_balance_label",
                "injury_risk_score","injury_risk_label","injury_event"]].copy()
    disp["date"] = disp["date"].dt.strftime("%b %d")
    st.dataframe(disp, width='stretch', hide_index=True)
    st.download_button("⬇ Download CSV", disp.to_csv(index=False), "wellness_data.csv","text/csv")

st.markdown("---")

st.markdown("### 📄 Download Full Report")
st.markdown("Generate and download a comprehensive PDF report with all charts, metrics, and interpretations.")

if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating PDF report..."):
        pdf_data = generate_pdf_report(df, raw)
        if pdf_data:
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_data,
                file_name="athlete_wellness_report.pdf",
                mime="application/pdf",
                help="Click to download the full analytics report as PDF"
            )
            st.success("PDF report generated! Click the download button above.")
        else:
            st.error("Failed to generate PDF report.")

st.caption("Athlete Wellness & Mood Analytics · Synthetic data (Synthea + PhysioNet-inspired) · Built with Python, Streamlit & Plotly")
