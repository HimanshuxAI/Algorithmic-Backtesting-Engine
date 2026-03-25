"""
charts.py — All matplotlib visualizations
Produces: equity curve, drawdown, returns distribution, trade analytics,
          monthly heatmap, rolling metrics, Monte Carlo fan chart.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── THEME ──────────────────────────────────────────────────────────
DARK_BG   = "#0a0a0b"
SURFACE   = "#111114"
BORDER    = "#1e1e24"
TEXT      = "#e8e8ec"
MUTED     = "#5a5a6e"
ACCENT    = "#c8f542"
RED       = "#ff4d4d"
BLUE      = "#7dd3fc"
ORANGE    = "#f97316"

def _set_theme():
    plt.rcParams.update({
        "figure.facecolor":   DARK_BG,
        "axes.facecolor":     SURFACE,
        "axes.edgecolor":     BORDER,
        "axes.labelcolor":    MUTED,
        "axes.titlecolor":    TEXT,
        "axes.grid":          True,
        "axes.grid.which":    "both",
        "grid.color":         BORDER,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.6,
        "text.color":         TEXT,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.facecolor":   SURFACE,
        "legend.edgecolor":   BORDER,
        "legend.labelcolor":  MUTED,
        "legend.fontsize":    8,
        "font.family":        "monospace",
        "lines.linewidth":    1.5,
    })


# ─────────────────────────────────────────────
#  MASTER DASHBOARD
# ─────────────────────────────────────────────
def plot_dashboard(
    equity_curve: pd.Series,
    trades_df:    pd.DataFrame,
    benchmark:    Optional[pd.Series] = None,
    summary:      dict                = None,
    strategy_name: str                = "Strategy",
    savepath:     str                 = "dashboard.png",
):
    _set_theme()
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.38,
        left=0.05, right=0.97,
        top=0.92,  bottom=0.07,
    )

    ax_equity  = fig.add_subplot(gs[0, :2])
    ax_dd      = fig.add_subplot(gs[1, :2])
    ax_dist    = fig.add_subplot(gs[0, 2])
    ax_monthly = fig.add_subplot(gs[1, 2])
    ax_rolling = fig.add_subplot(gs[2, :2])
    ax_scatter = fig.add_subplot(gs[2, 2])

    initial  = equity_curve.iloc[0]
    norm_eq  = equity_curve / initial * 100

    # ── 1. EQUITY CURVE ──────────────────────
    ax = ax_equity
    ax.plot(norm_eq.index, norm_eq.values, color=ACCENT, linewidth=1.8, zorder=3)
    ax.fill_between(norm_eq.index, 100, norm_eq.values,
                    where=norm_eq.values > 100,
                    alpha=0.12, color=ACCENT, zorder=2)
    ax.fill_between(norm_eq.index, 100, norm_eq.values,
                    where=norm_eq.values <= 100,
                    alpha=0.12, color=RED, zorder=2)

    if benchmark is not None:
        bm_norm = benchmark / benchmark.iloc[0] * 100
        bm_aligned = bm_norm.reindex(equity_curve.index, method="ffill").dropna()
        ax.plot(bm_aligned.index, bm_aligned.values,
                color=MUTED, linewidth=1.2, linestyle="--",
                label="Benchmark", alpha=0.7)

    ax.axhline(100, color=BORDER, linewidth=1, linestyle=":", zorder=1)

    # Mark entries/exits
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            if pd.notna(row.get("entry_date")):
                ed = pd.to_datetime(row["entry_date"])
                if ed in equity_curve.index:
                    ax.scatter(ed, norm_eq.loc[ed], marker="^",
                               color=ACCENT, s=25, zorder=4, alpha=0.7)
            if pd.notna(row.get("exit_date")):
                xd = pd.to_datetime(row["exit_date"])
                if xd in equity_curve.index:
                    clr = ACCENT if row.get("net_pnl", 0) > 0 else RED
                    ax.scatter(xd, norm_eq.loc[xd], marker="v",
                               color=clr, s=25, zorder=4, alpha=0.6)

    ax.set_title(f"EQUITY CURVE — {strategy_name}", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_ylabel("Indexed (base=100)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    _style_ax(ax)

    # stats annotation
    if summary:
        ann = (f"CAGR {summary.get('cagr_pct',0):.1f}% | "
               f"Sharpe {summary.get('sharpe_ratio',0):.2f} | "
               f"MaxDD {summary.get('max_drawdown_pct',0):.1f}%")
        ax.set_xlabel(ann, labelpad=4, color=MUTED)

    # ── 2. DRAWDOWN ──────────────────────────
    ax = ax_dd
    dd = (equity_curve - equity_curve.cummax()) / equity_curve.cummax() * 100
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5, zorder=2)
    ax.plot(dd.index, dd.values, color=RED, linewidth=1, zorder=3)
    ax.axhline(0, color=BORDER, linewidth=0.8)
    ax.set_title("UNDERWATER EQUITY (DRAWDOWN %)", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_ylabel("Drawdown %", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    _style_ax(ax)

    # ── 3. RETURN DISTRIBUTION ───────────────
    ax = ax_dist
    if not trades_df.empty and "return_pct" in trades_df.columns:
        rets = trades_df["return_pct"].dropna()
        bins = min(30, max(10, len(rets) // 3))
        wins  = rets[rets >= 0]
        losss = rets[rets <  0]
        ax.hist(wins,  bins=bins, color=ACCENT, alpha=0.75, label="Win",  edgecolor=DARK_BG, linewidth=0.3)
        ax.hist(losss, bins=bins, color=RED,    alpha=0.75, label="Loss", edgecolor=DARK_BG, linewidth=0.3)
        ax.axvline(rets.mean(), color=BLUE, linewidth=1.5, linestyle="--", label=f"Mean {rets.mean():.2f}%")
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title("TRADE RETURN DIST.", fontsize=9, color=TEXT,
                     loc="left", fontweight="bold", pad=8)
        ax.set_xlabel("Return %", fontsize=8)
        ax.legend(fontsize=7)
    _style_ax(ax)

    # ── 4. MONTHLY RETURNS HEATMAP ───────────
    ax = ax_monthly
    _plot_monthly_heatmap(ax, equity_curve)

    # ── 5. ROLLING SHARPE ────────────────────
    ax = ax_rolling
    daily_ret = equity_curve.pct_change().dropna()
    rf_daily  = (1 + 0.065) ** (1/252) - 1
    roll_win  = min(63, len(daily_ret) // 3)
    if roll_win > 5:
        roll_sharpe = (
            (daily_ret - rf_daily)
            .rolling(roll_win)
            .mean()
            .div(daily_ret.rolling(roll_win).std() + 1e-9)
            * np.sqrt(252)
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, color=BLUE, linewidth=1.2)
        ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                        where=roll_sharpe.values > 0, alpha=0.15, color=BLUE)
        ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                        where=roll_sharpe.values <= 0, alpha=0.15, color=RED)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axhline(2, color=ACCENT, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title(f"ROLLING SHARPE ({roll_win}d)", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    _style_ax(ax)

    # ── 6. TRADE SCATTER (hold days vs return) ──
    ax = ax_scatter
    if not trades_df.empty and "hold_days" in trades_df.columns:
        td = trades_df.dropna(subset=["hold_days", "return_pct"])
        colors = [ACCENT if p > 0 else RED for p in td["return_pct"]]
        ax.scatter(td["hold_days"], td["return_pct"],
                   c=colors, alpha=0.6, s=20, edgecolors="none")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xlabel("Hold Days", fontsize=8)
        ax.set_ylabel("Return %", fontsize=8)
    ax.set_title("HOLD DAYS vs RETURN", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    _style_ax(ax)

    # ── HEADER ──────────────────────────────
    fig.text(0.05, 0.975, "ALGORITHMIC BACKTESTING ENGINE",
             color=TEXT, fontsize=11, fontweight="bold",
             verticalalignment="top", fontfamily="monospace")
    fig.text(0.97, 0.975, f"© {strategy_name}",
             color=MUTED, fontsize=8, ha="right",
             verticalalignment="top", fontfamily="monospace")

    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"[charts] Dashboard saved → {savepath}")


# ─────────────────────────────────────────────
#  MONTHLY HEATMAP HELPER
# ─────────────────────────────────────────────
def _plot_monthly_heatmap(ax, equity_curve: pd.Series):
    daily_ret = equity_curve.pct_change().dropna()
    if daily_ret.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        return

    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.columns = [pd.Period(year=2000, month=m, freq="M").strftime("%b") for m in pivot.columns]

    # Custom diverging colormap (red → black → green)
    cmap = LinearSegmentedColormap.from_list(
        "rg", [(0.9, 0.15, 0.15), (0.07, 0.07, 0.09), (0.49, 0.93, 0.20)]
    )
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 1)

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Cell annotations
    for yi in range(pivot.shape[0]):
        for xi in range(pivot.shape[1]):
            v = pivot.values[yi, xi]
            if np.isfinite(v):
                txt_color = TEXT if abs(v) > vmax * 0.4 else MUTED
                ax.text(xi, yi, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color, fontfamily="monospace")

    ax.set_title("MONTHLY RETURNS %", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.grid(False)


# ─────────────────────────────────────────────
#  STRATEGY COMPARISON
# ─────────────────────────────────────────────
def plot_strategy_comparison(scorecard_df: pd.DataFrame, savepath: str = "strategy_comparison.png"):
    if scorecard_df is None or scorecard_df.empty:
        return

    _set_theme()
    ranked = scorecard_df.sort_values("resilience_score", ascending=True).copy()
    ranked["max_drawdown_abs"] = ranked["max_drawdown_pct"].abs()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)

    ax = axes[0]
    bar_colors = [BLUE if i < len(ranked) - 1 else ACCENT for i in range(len(ranked))]
    ax.barh(ranked["strategy"], ranked["resilience_score"], color=bar_colors, alpha=0.8)
    ax.set_title("RESILIENCE SCORE LEADERBOARD", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlabel("Composite Score", fontsize=8)
    for idx, value in enumerate(ranked["resilience_score"].tolist()):
        ax.text(value + 0.8, idx, f"{value:.1f}", va="center", color=MUTED, fontsize=8)
    _style_ax(ax)

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("score_map", [RED, ORANGE, ACCENT])
    bubble_size = np.clip(ranked["total_return_pct"].abs(), 5, 80) * 8
    scatter = ax.scatter(
        ranked["max_drawdown_abs"],
        ranked["sharpe_ratio"],
        s=bubble_size,
        c=ranked["resilience_score"],
        cmap=cmap,
        alpha=0.85,
        edgecolors=DARK_BG,
        linewidths=0.5,
    )
    for _, row in ranked.iterrows():
        ax.text(
            row["max_drawdown_abs"] + 0.3,
            row["sharpe_ratio"] + 0.02,
            row["strategy"],
            fontsize=7,
            color=MUTED,
        )
    ax.set_title("DRAWDOWN VS SHARPE", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlabel("Absolute Max Drawdown %", fontsize=8)
    ax.set_ylabel("Sharpe Ratio", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(BORDER)
    cbar.set_label("Resilience Score", color=MUTED, fontsize=8)
    _style_ax(ax)

    fig.suptitle("STRATEGY COMPARISON — Strength, Stability, and Risk",
                 color=TEXT, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"[charts] Strategy comparison saved → {savepath}")


# ─────────────────────────────────────────────
#  STRATEGY LAB SUMMARY
# ─────────────────────────────────────────────
def plot_strategy_lab(summary_df: pd.DataFrame, savepath: str = "strategy_lab.png"):
    if summary_df is None or summary_df.empty:
        return

    _set_theme()
    ranked = summary_df.sort_values("test_score", ascending=True).copy()
    for col in ["train_score", "test_score", "test_total_return_pct", "test_sharpe_ratio", "candidate_count"]:
        if col not in ranked.columns:
            ranked[col] = 0.0
    y_pos = np.arange(len(ranked))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)

    ax = axes[0]
    ax.barh(y_pos - 0.18, ranked["train_score"], height=0.32, color=BLUE, alpha=0.75, label="Train score")
    ax.barh(y_pos + 0.18, ranked["test_score"], height=0.32, color=ACCENT, alpha=0.85, label="Test score")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked["family"])
    ax.set_xlabel("Score", fontsize=8)
    ax.set_title("TRAIN VS TEST PROMOTION SCORE", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    for idx, row in enumerate(ranked.itertuples()):
        ax.text(row.test_score + 0.8, idx + 0.18, f"{row.test_score:.1f}", color=MUTED, va="center", fontsize=7)
    ax.legend(fontsize=8)
    _style_ax(ax)

    ax = axes[1]
    bubble_size = np.clip(ranked["candidate_count"].fillna(1), 1, 10) * 90
    scatter = ax.scatter(
        ranked["test_total_return_pct"],
        ranked["test_sharpe_ratio"],
        s=bubble_size,
        c=ranked["test_score"],
        cmap=LinearSegmentedColormap.from_list("lab_map", [RED, ORANGE, ACCENT]),
        alpha=0.85,
        edgecolors=DARK_BG,
        linewidths=0.6,
    )
    for row in ranked.itertuples():
        ax.text(
            row.test_total_return_pct + 0.25,
            row.test_sharpe_ratio + 0.02,
            row.family,
            fontsize=7,
            color=MUTED,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Test Return %", fontsize=8)
    ax.set_ylabel("Test Sharpe", fontsize=8)
    ax.set_title("HOLDOUT PERFORMANCE MAP", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(BORDER)
    cbar.set_label("Test Score", color=MUTED, fontsize=8)
    _style_ax(ax)

    fig.suptitle("STRATEGY LAB — Parameter Search and Promotion",
                 color=TEXT, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"[charts] Strategy lab chart saved → {savepath}")


# ─────────────────────────────────────────────
#  MONTE CARLO FAN CHART
# ─────────────────────────────────────────────
def plot_monte_carlo(
    mc_result: dict,
    initial_capital: float,
    savepath: str = "monte_carlo.png",
):
    _set_theme()
    if not mc_result or "all_final_values" not in mc_result:
        print("[charts] No MC data to plot.")
        return

    fv   = mc_result["all_final_values"]
    mdd  = mc_result["all_max_dds"]
    percentile_paths = mc_result.get("percentile_paths")
    sample_paths = mc_result.get("sample_paths")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK_BG)

    # Fan chart
    ax = axes[0]
    if percentile_paths:
        x = np.arange(1, len(percentile_paths["p50"]) + 1)
        if sample_paths is not None:
            for path in sample_paths[:20]:
                ax.plot(x, path / 1e5, color=BLUE, alpha=0.08, linewidth=0.8)
        ax.fill_between(
            x,
            percentile_paths["p10"] / 1e5,
            percentile_paths["p90"] / 1e5,
            color=ACCENT,
            alpha=0.16,
            label="10-90 pct band",
        )
        ax.plot(x, percentile_paths["p50"] / 1e5, color=ACCENT, linewidth=1.8, label="Median path")
        ax.axhline(initial_capital / 1e5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_title("BOOTSTRAP EQUITY FAN", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlabel("Trade Number", fontsize=8)
    ax.set_ylabel("Equity (₹ Lakhs)", fontsize=8)
    ax.legend(fontsize=7)
    _style_ax(ax)

    # Final value distribution
    ax = axes[1]
    ax.hist(fv / 1e5, bins=60, color=ACCENT, alpha=0.7, edgecolor=DARK_BG, linewidth=0.3)
    ax.axvline(initial_capital / 1e5, color=RED, linewidth=1.5,
               linestyle="--", label=f"Initial ₹{initial_capital/1e5:.1f}L")
    ax.axvline(np.median(fv) / 1e5, color=BLUE, linewidth=1.5,
               linestyle="--", label=f"Median ₹{np.median(fv)/1e5:.1f}L")
    ax.axvline(np.percentile(fv, 5)  / 1e5, color=RED, linewidth=1, linestyle=":", alpha=0.7)
    ax.axvline(np.percentile(fv, 95) / 1e5, color=ACCENT, linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title("FINAL EQUITY DISTRIBUTION", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlabel("Final Value (₹ Lakhs)", fontsize=8)
    ax.legend(fontsize=8)
    _style_ax(ax)

    # Max drawdown distribution
    ax = axes[2]
    ax.hist(mdd * 100, bins=60, color=RED, alpha=0.7, edgecolor=DARK_BG, linewidth=0.3)
    ax.axvline(np.median(mdd) * 100, color=BLUE, linewidth=1.5,
               linestyle="--", label=f"Median {np.median(mdd)*100:.1f}%")
    ax.axvline(np.percentile(mdd, 5) * 100, color=RED, linewidth=1, linestyle=":",
               alpha=0.7, label=f"5th pct {np.percentile(mdd,5)*100:.1f}%")
    ax.set_title("MAX DRAWDOWN DISTRIBUTION", fontsize=9, color=TEXT,
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlabel("Max Drawdown %", fontsize=8)
    ax.legend(fontsize=8)
    _style_ax(ax)

    fig.suptitle(f"MONTE CARLO SIMULATION — {mc_result.get('n_simulations', 0):,} Bootstrap Trials",
                 color=TEXT, fontsize=10, fontweight="bold", y=1.01)

    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"[charts] Monte Carlo chart saved → {savepath}")


# ─────────────────────────────────────────────
#  WALK-FORWARD CHART
# ─────────────────────────────────────────────
def plot_walk_forward(wf_df: pd.DataFrame, savepath: str = "walk_forward.png"):
    if wf_df.empty:
        return
    _set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG)

    metrics = [
        ("total_return_pct", "Return Per Fold %",  ACCENT),
        ("sharpe",           "Sharpe Per Fold",     BLUE),
        ("max_dd_pct",       "Max DD Per Fold %",   RED),
        ("win_rate_pct",     "Win Rate Per Fold %", ORANGE),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        vals = wf_df[col].values
        bars = ax.bar(wf_df["fold"], vals, color=color, alpha=0.75,
                      edgecolor=DARK_BG, linewidth=0.5)
        ax.axhline(vals.mean(), color=color, linewidth=1.5,
                   linestyle="--", alpha=0.8, label=f"Mean: {vals.mean():.2f}")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(label, fontsize=9, color=TEXT, loc="left",
                     fontweight="bold", pad=8)
        ax.set_xlabel("Fold", fontsize=8)
        ax.legend(fontsize=8)
        _style_ax(ax)

    fig.suptitle("WALK-FORWARD VALIDATION — Out-Of-Sample Performance",
                 color=TEXT, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"[charts] Walk-forward chart saved → {savepath}")


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, length=3)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data import generate_synthetic
    from strategies import MACrossStrategy
    from engine import BacktestEngine
    from execution import RiskConfig
    from analytics import PerformanceMetrics

    df = generate_synthetic(756)
    s  = MACrossStrategy(fast=20, slow=50)
    e  = BacktestEngine(s, risk_config=RiskConfig(sizing_method="kelly_fractional"))
    r  = e.run(df)

    eq  = r.equity_curve["equity"]
    pm  = PerformanceMetrics(eq, r.trade_df)
    summ = pm.summary()

    plot_dashboard(eq, r.trade_df, summary=summ, strategy_name="MA Cross",
                   savepath="/tmp/test_dashboard.png")
    print("Done.")
