#!/usr/bin/env python3
"""
2×3 summary behavior figure for the ABA/ABB relational reasoning fMRI task.

Layout
------
Row 0 (accuracy bar plots):
  (0,0)  Overall accuracy per participant — legend is here only
  (0,1)  Accuracy by rule type (ABA / ABB), touching bars per participant
  (0,2)  Accuracy by transition (repeat / switch), touching bars per participant

Row 1 (RT distributions, correct trials only):
  (1,0)  One KDE per participant
  (1,1)  Aggregated KDE for ABA vs ABB
  (1,2)  Aggregated KDE for repeat vs switch

Usage
-----
uv run python summary_figure.py --data-root ../../../data --outdir ../behavior_results

Statistics (for figure caption)
--------------------------------
  (0,1)  Paired t-test: ABA accuracy vs ABB accuracy across subjects
  (0,2)  Paired t-test: repeat accuracy vs switch accuracy across subjects
  (1,1)  Independent-samples t-test (or Mann–Whitney): ABA RT vs ABB RT (correct trials, pooled)
  (1,2)  Independent-samples t-test (or Mann–Whitney): repeat RT vs switch RT (correct trials, pooled)
All p-values are printed to stdout; include relevant ones in the figure caption.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde


# ── palette ──────────────────────────────────────────────────────────────────

# Up to 8 participants; we have 3.
SUBJECT_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_sessions(data_root: Path) -> pd.DataFrame:
    csvs = sorted(data_root.glob("*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_root}")

    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).fillna(False)

    str_cols = ["block", "sid", "rule_type", "A_stim", "B_stim",
                "A_prime", "B_prime", "rule_sequence", "test_sequence",
                "correct_next_stim", "slot_mapping", "correct_stim",
                "response_stim", "sample_stim"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    for c in ["trial", "correct", "rt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype("Int64")

    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── data prep ────────────────────────────────────────────────────────────────

def add_transition(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'transition' column: 'repeat' or 'switch' (first trial → NaN)."""
    tmp = df.sort_values(["sid", "trial"]).copy()
    tmp["prev_rule"] = tmp.groupby("sid")["rule_type"].shift(1)
    same = (tmp["rule_type"] == tmp["prev_rule"]).fillna(False)
    tmp["transition"] = np.where(
        tmp["prev_rule"].isna(), None,
        np.where(same, "repeat", "switch"),
    )
    return tmp


# ── statistical helpers ───────────────────────────────────────────────────────

def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test; returns (t, p)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan, np.nan
    return stats.ttest_rel(a[mask], b[mask])


def indep_mwu(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann–Whitney U; returns (U, p)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# ── plotting helpers ──────────────────────────────────────────────────────────

def touching_bars(ax: plt.Axes,
                  groups: list[str],
                  subject_values: dict[str, list[float]],
                  colors: list[str],
                  ylabel: str,
                  title: str,
                  show_legend: bool = False,
                  sid_labels: list[str] | None = None,
                  ylim: tuple[float, float] = (0.0, 1.05),
                  chance: float | None = None) -> None:
    """
    Bar chart with groups on the x-axis and touching per-subject bars inside
    each group.

    subject_values: {subject_id: [value_for_group0, value_for_group1, ...]}
    """
    n_groups = len(groups)
    sids = list(subject_values.keys())
    n_subj = len(sids)

    bar_width = 0.6 / n_subj          # touching bars fill 0.6 of a group slot
    group_centers = np.arange(n_groups)

    for s_idx, sid in enumerate(sids):
        vals = subject_values[sid]
        offsets = (np.arange(n_subj) - (n_subj - 1) / 2) * bar_width
        x = group_centers + offsets[s_idx]
        label = (sid_labels[s_idx] if sid_labels else f"S{s_idx + 1}") if show_legend else None
        ax.bar(x, vals, width=bar_width, color=colors[s_idx],
               label=label, edgecolor="white", linewidth=0.3)

    if chance is not None:
        ax.axhline(chance, linestyle="--", linewidth=1, color="gray", zorder=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)

    if show_legend:
        ax.legend(frameon=False, fontsize=8)


def kde_with_median(ax: plt.Axes,
                    rt_values: np.ndarray,
                    color: str,
                    label: str,
                    bw_method: str | float = "scott") -> None:
    """Plot a smoothed KDE and mark the median with a vertical dashed line."""
    rt_values = rt_values[~np.isnan(rt_values)]
    if len(rt_values) < 3:
        return
    kde = gaussian_kde(rt_values, bw_method=bw_method)
    x = np.linspace(max(rt_values.min() - 0.1, 0), rt_values.max() + 0.1, 300)
    ax.plot(x, kde(x), color=color, label=label)
    ax.fill_between(x, kde(x), alpha=0.15, color=color)
    med = np.median(rt_values)
    ax.axvline(med, color=color, linestyle="--", linewidth=1,
               label=f"{label} median={med:.2f}s")


# ── main figure ───────────────────────────────────────────────────────────────

def make_summary_figure(df_main: pd.DataFrame, outpath: Path) -> None:
    # Sorted subjects → stable ordering across panels
    sids = sorted(df_main["sid"].dropna().unique())
    n_subj = len(sids)
    colors = SUBJECT_COLORS[:n_subj]
    short_labels = [f"S{i + 1}" for i in range(n_subj)]

    df_main = add_transition(df_main)
    df_correct = df_main.loc[df_main["correct"] == 1].copy()

    # ── compute per-subject statistics ───────────────────────────────────────

    # Overall accuracy
    overall_acc = {
        sid: float(df_main.loc[df_main["sid"] == sid, "correct"].mean())
        for sid in sids
    }

    # Accuracy by rule type
    rule_acc: dict[str, dict[str, float]] = {sid: {} for sid in sids}
    for sid in sids:
        g = df_main.loc[df_main["sid"] == sid]
        for rule in ["ABA", "ABB"]:
            rule_acc[sid][rule] = float(g.loc[g["rule_type"] == rule, "correct"].mean())

    # Accuracy by transition
    trans_acc: dict[str, dict[str, float]] = {sid: {} for sid in sids}
    df_trans = df_main.loc[df_main["transition"].notna()].copy()
    for sid in sids:
        g = df_trans.loc[df_trans["sid"] == sid]
        for tr in ["repeat", "switch"]:
            trans_acc[sid][tr] = float(g.loc[g["transition"] == tr, "correct"].mean())

    # ── statistics (printed for caption) ─────────────────────────────────────

    aba_acc = np.array([rule_acc[s]["ABA"] for s in sids])
    abb_acc = np.array([rule_acc[s]["ABB"] for s in sids])
    t_rule, p_rule = paired_t(aba_acc, abb_acc)

    rep_acc = np.array([trans_acc[s]["repeat"] for s in sids])
    swi_acc = np.array([trans_acc[s]["switch"] for s in sids])
    t_trans, p_trans = paired_t(rep_acc, swi_acc)

    rt_aba = df_correct.loc[df_correct["rule_type"] == "ABA", "rt"].dropna().to_numpy()
    rt_abb = df_correct.loc[df_correct["rule_type"] == "ABB", "rt"].dropna().to_numpy()
    u_rule_rt, p_rule_rt = indep_mwu(rt_aba, rt_abb)

    df_trans_correct = df_correct.loc[df_correct["transition"].notna()]
    rt_rep = df_trans_correct.loc[df_trans_correct["transition"] == "repeat", "rt"].dropna().to_numpy()
    rt_swi = df_trans_correct.loc[df_trans_correct["transition"] == "switch", "rt"].dropna().to_numpy()
    u_trans_rt, p_trans_rt = indep_mwu(rt_rep, rt_swi)

    print("\n── Statistics for figure caption ──────────────────────────────")
    print(f"  (0,1) Accuracy ABA vs ABB:     paired t({len(sids)-1}) = {t_rule:.3f}, p = {p_rule:.4f}")
    print(f"  (0,2) Accuracy repeat vs switch: paired t({len(sids)-1}) = {t_trans:.3f}, p = {p_trans:.4f}")
    print(f"  (1,1) RT ABA vs ABB (MWU):     U = {u_rule_rt:.1f}, p = {p_rule_rt:.4f}")
    print(f"  (1,2) RT repeat vs switch (MWU): U = {u_trans_rt:.1f}, p = {p_trans_rt:.4f}")
    print("───────────────────────────────────────────────────────────────\n")

    # ── figure layout ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    # ── (0,0) Overall accuracy per participant ────────────────────────────────
    ax = axes[0, 0]
    bar_w = 0.4
    gap = 0.2                           # gap between bars
    x_pos = np.arange(n_subj) * (bar_w + gap)
    for i, sid in enumerate(sids):
        ax.bar(x_pos[i], overall_acc[sid], color=colors[i],
               label=short_labels[i], width=bar_w)
    ax.axhline(0.25, linestyle="--", linewidth=1, color="gray", zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels)
    # Extra room on the right for the legend
    ax.set_xlim(-0.4, x_pos[-1] + bar_w * 0.5 + 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, title="Participant",
              loc="center right")

    # ── (0,1) Accuracy by rule type ───────────────────────────────────────────
    ax = axes[0, 1]
    rule_subject_vals = {
        sid: [rule_acc[sid]["ABA"], rule_acc[sid]["ABB"]]
        for sid in sids
    }
    touching_bars(
        ax=ax,
        groups=["ABA", "ABB"],
        subject_values=rule_subject_vals,
        colors=colors,
        ylabel="Accuracy",
        title="Accuracy by rule type",
        show_legend=False,
        ylim=(0.0, 1.05),
        chance=0.25,
    )

    # ── (0,2) Accuracy by transition ─────────────────────────────────────────
    ax = axes[0, 2]
    trans_subject_vals = {
        sid: [trans_acc[sid]["repeat"], trans_acc[sid]["switch"]]
        for sid in sids
    }
    touching_bars(
        ax=ax,
        groups=["Repeat", "Switch"],
        subject_values=trans_subject_vals,
        colors=colors,
        ylabel="Accuracy",
        title="Accuracy by trial transition",
        show_legend=False,
        ylim=(0.0, 1.05),
        chance=0.25,
    )

    # ── (1,0) RT distribution per participant (correct trials) ────────────────
    ax = axes[1, 0]
    for i, sid in enumerate(sids):
        rt_subj = df_correct.loc[df_correct["sid"] == sid, "rt"].dropna().to_numpy()
        kde_with_median(ax, rt_subj, colors[i], short_labels[i])
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT distribution (correct trials)")
    ax.legend(frameon=False, fontsize=7)

    # ── (1,1) RT distribution by rule type (aggregated) ──────────────────────
    ax = axes[1, 1]
    for rule, color in [("ABA", "#4477AA"), ("ABB", "#EE6677")]:
        rt_vals = df_correct.loc[df_correct["rule_type"] == rule, "rt"].dropna().to_numpy()
        kde_with_median(ax, rt_vals, color, rule)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT by rule type (aggregated)")
    ax.legend(frameon=False, fontsize=7)

    # ── (1,2) RT distribution by transition (aggregated) ─────────────────────
    ax = axes[1, 2]
    for tr, color in [("repeat", "#228833"), ("switch", "#CCBB44")]:
        rt_vals = df_trans_correct.loc[df_trans_correct["transition"] == tr, "rt"].dropna().to_numpy()
        kde_with_median(ax, rt_vals, color, tr)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT by trial transition (aggregated)")
    ax.legend(frameon=False, fontsize=7)

    # ── panel letters ─────────────────────────────────────────────────────────
    for ax, letter in zip(axes.flat, "ABCDEF"):
        ax.text(-0.2, 1.07, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="2×3 summary behavior figure")
    parser.add_argument("--data-root", required=True,
                        help="Root data directory containing session subfolders")
    parser.add_argument("--outdir", required=True,
                        help="Directory to write the figure")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_sessions(data_root)
    df_main = df.loc[df["block"].astype(str).str.startswith("main_run")].copy()

    if len(df_main) == 0:
        raise ValueError("No main-task trials found.")

    make_summary_figure(df_main, outdir / "fig_summary.png")


if __name__ == "__main__":
    main()
