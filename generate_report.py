import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_CSV = os.path.join(BASE_DIR, 'analytics_pipeline_output.csv')
RAW_CSV = os.path.join(BASE_DIR, 'raw_wearable_data.csv')
OUTPUT_PDF = os.path.join(BASE_DIR, 'user_report.pdf')

pipeline = pd.read_csv(PIPELINE_CSV, parse_dates=['date'])
raw = pd.read_csv(RAW_CSV, parse_dates=['timestamp']) if os.path.exists(RAW_CSV) else None

if pipeline.empty:
    raise SystemExit('No pipeline data found for report generation.')

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

with PdfPages(OUTPUT_PDF) as pdf:
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
    # Show every 7th date to reduce crowding
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

print(f'Report saved to: {OUTPUT_PDF}')
