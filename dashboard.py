"""
TASK 9: Analytics Dashboard
============================
Generates:
1. Readiness trend chart
2. Sleep vs Recovery chart
3. Training load balance (ACWR)
4. Injury risk timeline
5. Combined HTML dashboard
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Load pipeline output ──────────────────────────────────────────
df = pd.read_csv("/analytics_pipeline_output.csv")
df["date"] = pd.to_datetime(df["date"])

COLORS = {
    "high":        "#22c55e",
    "moderate":    "#f59e0b",
    "low":         "#ef4444",
    "optimal":     "#3b82f6",
    "partial":     "#a78bfa",
    "poor":        "#f87171",
    "bg":          "#0f172a",
    "surface":     "#1e293b",
    "text":        "#e2e8f0",
    "muted":       "#94a3b8",
    "accent":      "#38bdf8",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["surface"],
    "axes.edgecolor":    COLORS["muted"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["muted"],
    "ytick.color":       COLORS["muted"],
    "text.color":        COLORS["text"],
    "grid.color":        "#334155",
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

dates = df["date"]
x = range(len(df))

# ─────────────────────────────────────────────────────────────────
# FIGURE 1: READINESS TREND
# ─────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["bg"])
ax.fill_between(x, df["readiness_score"], alpha=0.15, color=COLORS["accent"])
ax.plot(x, df["readiness_score"], color=COLORS["accent"], linewidth=2, label="Readiness Score")
# Colour-code points by label
for label, color in [("high", COLORS["high"]), ("moderate", COLORS["moderate"]), ("low", COLORS["low"])]:
    mask = df["readiness_label"] == label
    ax.scatter(np.array(list(x))[mask], df["readiness_score"][mask], color=color, s=40, zorder=5)
ax.axhline(75, color=COLORS["high"],    linestyle="--", alpha=0.5, linewidth=1, label="High threshold (75)")
ax.axhline(50, color=COLORS["moderate"],linestyle="--", alpha=0.5, linewidth=1, label="Moderate threshold (50)")
ax.set_xlim(0, len(df)-1)
ax.set_ylim(0, 105)
ax.set_xticks(range(0, len(df), 5))
ax.set_xticklabels([df["date"].iloc[i].strftime("%b %d") for i in range(0, len(df), 5)], rotation=30)
ax.set_ylabel("Score (0–100)")
ax.set_title("Readiness Score Trend", fontsize=13, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
ax.grid(True, axis="y")
# Mark injury events
for i, row in df[df["injury_event"] == 1].iterrows():
    ax.axvline(i, color="#f43f5e", alpha=0.6, linewidth=1.5, linestyle=":")
    ax.text(i, 102, "⚡", ha="center", fontsize=9, color="#f43f5e")
plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/chart1_readiness.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────
# FIGURE 2: SLEEP vs RECOVERY
# ─────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor=COLORS["bg"], sharex=True)
fig2.subplots_adjust(hspace=0.1)

ax1, ax2 = axes
ax1.bar(x, df["sleep_duration_h"], color=COLORS["accent"], alpha=0.7, label="Sleep (h)")
ax1.axhline(7.5, color=COLORS["high"], linestyle="--", alpha=0.6, linewidth=1, label="Target: 7.5h")
ax1.axhline(6.0, color=COLORS["low"], linestyle="--", alpha=0.6, linewidth=1, label="Minimum: 6.0h")
ax1.set_ylim(0, 10)
ax1.set_ylabel("Hours")
ax1.legend(loc="upper right", fontsize=8, framealpha=0.2)
ax1.set_title("Sleep Duration vs Recovery State", fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, axis="y")

recovery_colors = {"optimal": COLORS["high"], "partial": COLORS["moderate"], "poor": COLORS["low"]}
bar_colors = [recovery_colors.get(s, COLORS["muted"]) for s in df["recovery_state"]]
ax2.bar(x, [1]*len(df), color=bar_colors, alpha=0.85)
ax2.set_yticks([])
ax2.set_ylabel("Recovery")
ax2.set_xticks(range(0, len(df), 5))
ax2.set_xticklabels([df["date"].iloc[i].strftime("%b %d") for i in range(0, len(df), 5)], rotation=30)
patches = [mpatches.Patch(color=v, label=k) for k, v in recovery_colors.items()]
ax2.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.2)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/chart2_sleep_recovery.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────
# FIGURE 3: TRAINING LOAD BALANCE (ACWR)
# ─────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(2, 1, figsize=(12, 5.5), facecolor=COLORS["bg"], sharex=True)
fig3.subplots_adjust(hspace=0.15)

ax1, ax2 = axes
ax1.plot(x, df["acute_load"],   color="#fb923c", linewidth=2, label="Acute Load (7d avg)")
ax1.plot(x, df["chronic_load"], color="#60a5fa", linewidth=2, label="Chronic Load (28d avg)")
ax1.fill_between(x, df["acute_load"], df["chronic_load"], alpha=0.1, color="#a78bfa")
ax1.set_ylabel("Load (AU)")
ax1.legend(fontsize=8, framealpha=0.2)
ax1.set_title("Training Load Balance : Acute:Chronic Workload Ratio", fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, axis="y")

acwr_colors = [
    COLORS["low"] if v < 0.8 else
    COLORS["high"] if v <= 1.3 else
    COLORS["moderate"] if v <= 1.5 else "#dc2626"
    for v in df["acwr"].fillna(1)
]
ax2.bar(x, df["acwr"].fillna(1), color=acwr_colors, alpha=0.85)
ax2.axhline(0.8, color=COLORS["low"],  linestyle="--", alpha=0.6, linewidth=1)
ax2.axhline(1.3, color=COLORS["high"], linestyle="--", alpha=0.6, linewidth=1)
ax2.axhline(1.5, color="#dc2626",      linestyle="--", alpha=0.6, linewidth=1)
ax2.set_ylabel("ACWR")
ax2.set_xticks(range(0, len(df), 5))
ax2.set_xticklabels([df["date"].iloc[i].strftime("%b %d") for i in range(0, len(df), 5)], rotation=30)
ax2.grid(True, axis="y")
patches = [
    mpatches.Patch(color=COLORS["low"],  label="Undertraining (<0.8)"),
    mpatches.Patch(color=COLORS["high"], label="Optimal (0.8–1.3)"),
    mpatches.Patch(color=COLORS["moderate"], label="High (1.3–1.5)"),
    mpatches.Patch(color="#dc2626",      label="Overreaching (>1.5)"),
]
ax2.legend(handles=patches, fontsize=8, framealpha=0.2, loc="upper right")
plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/chart3_load_balance.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────
# FIGURE 4: INJURY RISK TIMELINE
# ─────────────────────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["bg"])
risk_colors = [
    COLORS["low"] if r == "low" else
    COLORS["moderate"] if r == "moderate" else COLORS["low"]
    for r in df["injury_risk_label"]
]
bars = ax.bar(x, df["injury_risk_score"].fillna(0), color=risk_colors, alpha=0.8)
ax.plot(x, df["injury_risk_score"].fillna(0), color="white", linewidth=1, alpha=0.4)
# Mark actual injury events
for i, row in df[df["injury_event"] == 1].iterrows():
    ax.scatter(i, df["injury_risk_score"].iloc[i] + 3, marker="v", color="#f43f5e", s=80, zorder=6)
ax.axhline(60, color=COLORS["low"],      linestyle="--", alpha=0.6, linewidth=1.5, label="High risk threshold (60)")
ax.axhline(30, color=COLORS["moderate"], linestyle="--", alpha=0.6, linewidth=1.5, label="Moderate threshold (30)")
ax.set_ylim(0, 110)
ax.set_xlim(0, len(df)-1)
ax.set_xticks(range(0, len(df), 5))
ax.set_xticklabels([df["date"].iloc[i].strftime("%b %d") for i in range(0, len(df), 5)], rotation=30)
ax.set_ylabel("Injury Risk Score (0–100)")
ax.set_title("Injury Risk Score Timeline  (▼ = actual injury event)", fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=8, framealpha=0.2)
ax.grid(True, axis="y")
plt.tight_layout()
fig4.savefig("/mnt/user-data/outputs/chart4_injury_risk.png", dpi=150, bbox_inches="tight")
plt.close()

print("✅ All 4 charts saved to /mnt/user-data/outputs/")

# ─────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────────
print("\n── PIPELINE SUMMARY ──────────────────────────────────────")
print(f"Days analyzed:          {len(df)}")
print(f"Avg readiness score:    {df['readiness_score'].mean():.1f}")
print(f"Recovery distribution:  {df['recovery_state'].value_counts().to_dict()}")
print(f"Load balance labels:    {df['load_balance_label'].value_counts().to_dict()}")
print(f"Injury events:          {df['injury_event'].sum()}")
print(f"High-risk days:         {(df['injury_risk_label']=='high').sum()}")
print(f"ACWR range:             {df['acwr'].min():.2f} – {df['acwr'].max():.2f}")
