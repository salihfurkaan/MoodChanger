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
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
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

    insights = [
        f"Average readiness score of {avg_readiness:.1f}/100 indicates {'strong' if avg_readiness >= 75 else 'moderate' if avg_readiness >= 50 else 'low'} overall performance.",
        f"{high_readiness_days} high-readiness days suggest good training periods.",
        f"{optimal_recovery_days} optimal recovery days show effective rest management.",
        f"{injury_events} injury events and {high_injury_risk_days} high-risk days require attention.",
        f"ACWR average of {acwr_avg:.2f} shows {'optimal' if 0.8 <= acwr_avg <= 1.3 else 'concerning'} load balance."
    ]

    # Professional color scheme
    colors = {
        'primary': '#1a365d',    # Dark blue
        'secondary': '#2d3748',  # Gray
        'accent': '#3182ce',     # Blue
        'success': '#38a169',    # Green
        'warning': '#d69e2e',    # Yellow
        'danger': '#e53e3e',     # Red
        'light': '#f7fafc',      # Light gray
        'border': '#e2e8f0'      # Border gray
    }

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def add_header_footer(fig, page_num, total_pages):
        """Add professional header and footer to each page"""
        # Header
        fig.text(0.05, 0.95, 'MoodChanger Analytics', fontsize=12, fontweight='bold', color=colors['primary'])
        fig.text(0.95, 0.95, f'Page {page_num} of {total_pages}', fontsize=10, ha='right', color=colors['secondary'])
        
        # Footer
        fig.text(0.05, 0.05, 'Confidential - Athlete Wellness Report', fontsize=8, color=colors['secondary'])
        fig.text(0.95, 0.05, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', fontsize=8, ha='right', color=colors['secondary'])

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Cover Page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Background
        ax.add_patch(Rectangle((0,0),1,1, facecolor=colors['light'], edgecolor='none', zorder=-2))
        ax.add_patch(Rectangle((0,0.6),1,0.4, facecolor=colors['primary'], alpha=0.1, edgecolor='none', zorder=-1))
        
        # Title
        ax.text(0.5, 0.8, 'ATHLETE WELLNESS', ha='center', va='center', fontsize=36, fontweight='bold', color=colors['primary'])
        ax.text(0.5, 0.72, 'ANALYTICS REPORT', ha='center', va='center', fontsize=28, fontweight='bold', color=colors['accent'])
        
        # Period
        ax.text(0.5, 0.6, f'Analysis Period: {period}', ha='center', va='center', fontsize=16, color=colors['secondary'])
        
        # Summary metrics in a box
        ax.add_patch(Rectangle((0.2,0.2),0.6,0.35, facecolor='white', edgecolor=colors['border'], linewidth=2))
        
        metrics = [
            ('Total Days', len(pipeline)),
            ('Avg Readiness', f"{avg_readiness:.1f}/100"),
            ('High Readiness Days', high_readiness_days),
            ('Injury Events', injury_events),
            ('Avg ACWR', f"{acwr_avg:.2f}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            y_pos = 0.5 - i * 0.06
            ax.text(0.25, y_pos, label, fontsize=12, fontweight='bold', color=colors['primary'])
            ax.text(0.65, y_pos, str(value), fontsize=12, ha='right', color=colors['secondary'])
        
        add_header_footer(fig, 1, 5)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Executive Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'EXECUTIVE SUMMARY', ha='center', va='center', fontsize=24, fontweight='bold', color=colors['primary'])
        
        # Key insights - improved spacing and text handling
        y_pos = 0.82  # Moved up slightly for better positioning
        for insight in insights:
            # Split long insights into multiple lines if needed
            words = insight.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) < 80:  # Approximate character limit per line
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Print each line with proper spacing
            for line in lines:
                ax.text(0.1, y_pos, f'• {line}', fontsize=11, color=colors['secondary'])
                y_pos -= 0.05
            y_pos -= 0.02  # Extra space between insights
        
        # KPI Cards
        kpi_data = [
            ('Readiness Score', f"{avg_readiness:.1f}", colors['success'] if avg_readiness >= 75 else colors['warning']),
            ('Recovery Days', optimal_recovery_days, colors['success']),
            ('Injury Risk Days', high_injury_risk_days, colors['danger'] if high_injury_risk_days > 5 else colors['warning']),
            ('ACWR Balance', f"{acwr_avg:.2f}", colors['success'] if 0.8 <= acwr_avg <= 1.3 else colors['warning'])
        ]
        
        for i, (label, value, color) in enumerate(kpi_data):
            col = i % 2
            row = i // 2
            x_pos = 0.1 + col * 0.4
            y_pos = 0.4 - row * 0.15
            
            ax.add_patch(Rectangle((x_pos, y_pos), 0.35, 0.12, facecolor='white', edgecolor=color, linewidth=3))
            ax.text(x_pos + 0.175, y_pos + 0.09, label, ha='center', va='center', fontsize=14, fontweight='bold', color=colors['primary'])
            ax.text(x_pos + 0.175, y_pos + 0.04, value, ha='center', va='center', fontsize=18, fontweight='bold', color=color)
        
        add_header_footer(fig, 2, 5)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3: Readiness Analysis
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main readiness trend
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(pipeline['date'], pipeline['readiness_score'], color=colors['accent'], linewidth=3, label='Daily Readiness')
        ax1.plot(pipeline['date'], pipeline['readiness_score'].rolling(7, min_periods=1).mean(), color=colors['danger'], linewidth=2, linestyle='--', label='7-Day Trend')
        ax1.axhline(75, color=colors['success'], linestyle='-', linewidth=2, alpha=0.7, label='High Threshold')
        ax1.axhline(50, color=colors['warning'], linestyle='-', linewidth=2, alpha=0.7, label='Moderate Threshold')
        ax1.fill_between(pipeline['date'], pipeline['readiness_score'], alpha=0.1, color=colors['accent'])
        ax1.set_title('Readiness Score Trend', fontsize=16, fontweight='bold', color=colors['primary'], pad=20)
        ax1.set_ylabel('Score (0-100)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor=colors['border'])
        ax1.grid(True, alpha=0.3, color=colors['border'])
        # Fix overlapping x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        n = len(pipeline)
        step = max(1, n // 7)
        ax1.set_xticks(pipeline['date'][::step])
        ax1.set_xticklabels([d.strftime('%m-%d') for d in pipeline['date'][::step]])
        
        # Readiness distribution
        ax2 = fig.add_subplot(gs[1, 0])
        readiness_counts = pipeline['readiness_label'].value_counts()
        colors_pie = [colors['success'], colors['warning'], colors['danger']]
        ax2.pie(readiness_counts.values, labels=readiness_counts.index, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax2.set_title('Readiness Distribution', fontsize=14, fontweight='bold', color=colors['primary'])
        
        # HRV trend
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(pipeline['date'], pipeline['HRV_ms'], color=colors['secondary'], linewidth=2)
        ax3.set_title('HRV Trend (ms)', fontsize=14, fontweight='bold', color=colors['primary'])
        ax3.set_ylabel('HRV (ms)', fontsize=12)
        ax3.grid(True, alpha=0.3, color=colors['border'])
        # Fix overlapping x-axis labels
        ax3.tick_params(axis='x', rotation=45)
        n = len(pipeline)
        step = max(1, n // 7)
        ax3.set_xticks(pipeline['date'][::step])
        ax3.set_xticklabels([d.strftime('%m-%d') for d in pipeline['date'][::step]])
        
        add_header_footer(fig, 3, 5)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 4: Training Load & Recovery
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ACWR timeline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(pipeline['date'], pipeline['acwr'], color=colors['warning'], linewidth=3, marker='o', markersize=3, markerfacecolor='white', markeredgecolor=colors['warning'])
        ax1.axhline(1.3, color=colors['success'], linestyle='-', linewidth=2, label='Optimal Upper')
        ax1.axhline(1.5, color=colors['danger'], linestyle='-', linewidth=2, label='Overreaching')
        ax1.axhline(0.8, color=colors['accent'], linestyle='-', linewidth=2, label='Undertraining')
        ax1.fill_between(pipeline['date'], pipeline['acwr'], 1.5, where=(pipeline['acwr'] > 1.5), color=colors['danger'], alpha=0.2)
        ax1.set_title('Acute:Chronic Workload Ratio (ACWR)', fontsize=16, fontweight='bold', color=colors['primary'], pad=20)
        ax1.set_ylabel('ACWR', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor=colors['border'])
        ax1.grid(True, alpha=0.3, color=colors['border'])
        # Fix overlapping x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        n = len(pipeline)
        step = max(1, n // 7)
        ax1.set_xticks(pipeline['date'][::step])
        ax1.set_xticklabels([d.strftime('%m-%d') for d in pipeline['date'][::step]])
        
        # Recovery state distribution
        ax2 = fig.add_subplot(gs[1, 0])
        recovery_counts = pipeline['recovery_state'].value_counts()
        colors_pie_rec = [colors['success'], colors['warning'], colors['danger']]
        ax2.pie(recovery_counts.values, labels=recovery_counts.index, autopct='%1.1f%%', colors=colors_pie_rec, startangle=90)
        ax2.set_title('Recovery State Distribution', fontsize=14, fontweight='bold', color=colors['primary'])
        
        # Sleep duration trend
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.bar(pipeline['date'], pipeline['sleep_duration_h'], color=colors['accent'], alpha=0.7)
        ax3.axhline(7.5, color=colors['success'], linestyle='--', linewidth=2, label='Target')
        ax3.set_title('Sleep Duration (hours)', fontsize=14, fontweight='bold', color=colors['primary'])
        ax3.set_ylabel('Hours', fontsize=12)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, color=colors['border'])
        # Fix overlapping x-axis labels
        ax3.tick_params(axis='x', rotation=45)
        n = len(pipeline)
        step = max(1, n // 7)
        ax3.set_xticks(pipeline['date'][::step])
        ax3.set_xticklabels([d.strftime('%m-%d') for d in pipeline['date'][::step]])
        
        add_header_footer(fig, 4, 5)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 5: Injury Risk & Raw Data
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Injury risk timeline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(pipeline['date'], pipeline['injury_risk_score'], color=colors['danger'], alpha=0.7, edgecolor=colors['danger'], linewidth=0.5)
        ax1.axhline(60, color=colors['danger'], linestyle='-', linewidth=2, label='High Risk')
        ax1.axhline(30, color=colors['warning'], linestyle='-', linewidth=2, label='Moderate Risk')
        ax1.set_title('Injury Risk Score Timeline', fontsize=16, fontweight='bold', color=colors['primary'], pad=20)
        ax1.set_ylabel('Risk Score (0-100)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor=colors['border'])
        ax1.grid(True, alpha=0.3, color=colors['border'])
        # Fix overlapping x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        n = len(pipeline)
        step = max(1, n // 7)
        ax1.set_xticks(pipeline['date'][::step])
        ax1.set_xticklabels([d.strftime('%m-%d') for d in pipeline['date'][::step]])
        
        # ACWR vs Soreness correlation
        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(pipeline['acwr'], pipeline['soreness_score'], c=pipeline['injury_risk_score'], cmap='RdYlGn_r', s=50, alpha=0.8, edgecolors=colors['border'])
        ax2.set_title('ACWR vs Soreness Correlation', fontsize=14, fontweight='bold', color=colors['primary'])
        ax2.set_xlabel('ACWR', fontsize=12)
        ax2.set_ylabel('Soreness (1-10)', fontsize=12)
        cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label('Injury Risk', fontsize=10)
        ax2.grid(True, alpha=0.3, color=colors['border'])
        
        # Raw sensor data (if available)
        if raw is not None and not raw.empty:
            sample_day = raw['timestamp'].dt.date.max()
            sample = raw[raw['timestamp'].dt.date == sample_day]
            if not sample.empty:
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.plot(sample['timestamp'], sample['HR_bpm'], color=colors['accent'], linewidth=2, label='HR')
                ax3.set_title(f'Raw HR Signal - {sample_day}', fontsize=14, fontweight='bold', color=colors['primary'])
                ax3.set_ylabel('HR (bpm)', fontsize=12)
                ax3.legend(loc='upper right', fontsize=10)
                ax3.grid(True, alpha=0.3, color=colors['border'])
                # Fix overlapping x-axis labels for timestamps
                ax3.tick_params(axis='x', rotation=45)
                n_timestamps = len(sample)
                step_ts = max(1, n_timestamps // 6)  # Show about 6 time points
                ax3.set_xticks(sample['timestamp'][::step_ts])
                ax3.set_xticklabels([t.strftime('%H:%M') for t in sample['timestamp'][::step_ts]])
            else:
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.text(0.5, 0.5, 'No raw data available', ha='center', va='center', fontsize=12, color=colors['secondary'])
                ax3.set_title('Raw Sensor Data', fontsize=14, fontweight='bold', color=colors['primary'])
                ax3.axis('off')
        else:
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.text(0.5, 0.5, 'No raw data available', ha='center', va='center', fontsize=12, color=colors['secondary'])
            ax3.set_title('Raw Sensor Data', fontsize=14, fontweight='bold', color=colors['primary'])
            ax3.axis('off')
        
        add_header_footer(fig, 5, 5)
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
