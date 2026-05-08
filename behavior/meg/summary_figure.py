#!/usr/bin/env python3
"""
2×3 summary behavior figure for the ABA/ABB relational reasoning MEG task.

Layout
------
Row 0 (accuracy bar plots):
  (0,0)  Overall accuracy per participant — legend is here only
  (0,1)  Accuracy by rule type (ABA / ABB), touching bars per participant
  (0,2)  Accuracy by trial type (match / rule_order / random), touching bars per participant

Row 1 (RT distributions, correct trials only):
  (1,0)  One KDE per participant
  (1,1)  Aggregated KDE for ABA vs ABB rule types
  (1,2)  Aggregated KDE for match vs rule_order vs random trial types

Usage
-----
uv run python behavior/meg/summary_figure.py --data-dir data --outdir behavior/meg

Statistics (for figure caption)
--------------------------------
  (0,1)  Paired t-test: ABA accuracy vs ABB accuracy across subjects
  (0,2)  Pairwise Mann–Whitney U: accuracy by trial type (match/rule_order/random)
  (1,1)  Mann–Whitney U: ABA RT vs ABB RT (correct trials, pooled)
  (1,2)  Pairwise Mann–Whitney U: RT by trial type (correct trials, pooled)
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
from scipy import stats
from scipy.stats import gaussian_kde


# ── palette ──────────────────────────────────────────────────────────────────

# Up to 8 participants
SUBJECT_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
                  "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]

TRIAL_TYPE_COLORS = {
    "match":      "#4477AA",
    "rule_order": "#EE6677",
    "random":     "#228833",
}


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_sessions(data_dir: Path) -> pd.DataFrame:
    """Load all sub-*_events.csv files found under data_dir."""
    csvs = sorted(data_dir.glob("*/sub-*_events.csv"))
    if not csvs:
        raise FileNotFoundError(f"No sub-*_events.csv files found under {data_dir}")

    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    str_cols = ["block", "sid", "rule_type", "A_stim", "B_stim",
                "rule_sequence", "test_type", "X_stim", "Y_stim",
                "test_sequence", "mismatch_type"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")

    for c in ["trial", "correct", "rt", "match"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype("Int64")

    return df


def add_trial_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'trial_type' column: 'match', 'rule_order', or 'random'."""
    df = df.copy()
    df["trial_type"] = np.where(
        df["match"] == 1,
        "match",
        np.where(df["mismatch_type"] == "rule_order", "rule_order", "random"),
    )
    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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

    bar_width = 0.6 / n_subj
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

def make_summary_figure(df: pd.DataFrame, outpath: Path) -> None:
    df = add_trial_type(df)

    sids = sorted(df["sid"].dropna().unique())
    n_subj = len(sids)
    colors = SUBJECT_COLORS[:n_subj]
    short_labels = [f"S{i + 1}" for i in range(n_subj)]

    df_correct = df.loc[df["correct"] == 1].copy()

    # ── compute per-subject statistics ───────────────────────────────────────

    overall_acc = {
        sid: float(df.loc[df["sid"] == sid, "correct"].mean())
        for sid in sids
    }

    rule_acc: dict[str, dict[str, float]] = {sid: {} for sid in sids}
    for sid in sids:
        g = df.loc[df["sid"] == sid]
        for rule in ["ABA", "ABB"]:
            rule_acc[sid][rule] = float(g.loc[g["rule_type"] == rule, "correct"].mean())

    trial_type_acc: dict[str, dict[str, float]] = {sid: {} for sid in sids}
    for sid in sids:
        g = df.loc[df["sid"] == sid]
        for tt in ["match", "rule_order", "random"]:
            trial_type_acc[sid][tt] = float(
                g.loc[g["trial_type"] == tt, "correct"].mean()
            )

    # ── statistics (printed for caption) ─────────────────────────────────────

    # Accuracy: binomial tests vs chance (0.5) and chi-square comparisons
    n_total   = len(df)
    n_correct = int(df["correct"].sum())
    binom_overall = stats.binomtest(n_correct, n_total, p=0.5)

    rule_binom = {}
    for rule in ["ABA", "ABB"]:
        g = df.loc[df["rule_type"] == rule]
        nc, nt = int(g["correct"].sum()), len(g)
        rule_binom[rule] = (nc, nt, stats.binomtest(nc, nt, p=0.5))
    ct_rule = pd.crosstab(df["rule_type"], df["correct"])
    chi2_rule, p_chi2_rule, dof_rule, _ = stats.chi2_contingency(ct_rule)

    tt_binom = {}
    for tt in ["match", "rule_order", "random"]:
        g = df.loc[df["trial_type"] == tt]
        nc, nt = int(g["correct"].sum()), len(g)
        tt_binom[tt] = (nc, nt, stats.binomtest(nc, nt, p=0.5))
    ct_tt = pd.crosstab(df["trial_type"], df["correct"])
    chi2_tt, p_chi2_tt, dof_tt, _ = stats.chi2_contingency(ct_tt)
    tt_fisher = {}
    for a, b in [("match", "rule_order"), ("match", "random"), ("rule_order", "random")]:
        sub = df.loc[df["trial_type"].isin([a, b])]
        ct = pd.crosstab(sub["trial_type"], sub["correct"])
        _, p_fish = stats.fisher_exact(ct)
        tt_fisher[(a, b)] = p_fish

    # RT: medians per participant, MWU comparisons
    rt_all = df_correct["rt"].dropna().to_numpy()
    rt_med_per_sid = {
        sid: float(np.median(df_correct.loc[df_correct["sid"] == sid, "rt"].dropna()))
        for sid in sids
    }

    rt_aba = df_correct.loc[df_correct["rule_type"] == "ABA", "rt"].dropna().to_numpy()
    rt_abb = df_correct.loc[df_correct["rule_type"] == "ABB", "rt"].dropna().to_numpy()
    u_rule_rt, p_rule_rt = indep_mwu(rt_aba, rt_abb)

    rt_match = df_correct.loc[df_correct["trial_type"] == "match",      "rt"].dropna().to_numpy()
    rt_ro    = df_correct.loc[df_correct["trial_type"] == "rule_order",  "rt"].dropna().to_numpy()
    rt_rand  = df_correct.loc[df_correct["trial_type"] == "random",      "rt"].dropna().to_numpy()
    u_match_ro,   p_match_ro   = indep_mwu(rt_match, rt_ro)
    u_match_rand, p_match_rand = indep_mwu(rt_match, rt_rand)
    u_ro_rand,    p_ro_rand    = indep_mwu(rt_ro, rt_rand)

    print("\n── Statistics for figure caption ──────────────────────────────")

    print(f"\n  Panel (0,0) — Overall accuracy")
    print(f"    {n_correct}/{n_total} = {n_correct/n_total:.4f}")
    print(f"    Binomial test vs chance (0.5): p = {binom_overall.pvalue:.4f}")

    print(f"\n  Panel (0,1) — Accuracy by rule type")
    for rule in ["ABA", "ABB"]:
        nc, nt, bt = rule_binom[rule]
        print(f"    {rule}: {nc}/{nt} = {nc/nt:.4f}, binomial p = {bt.pvalue:.4f}")
    print(f"    Chi-square (ABA vs ABB): chi2({dof_rule}) = {chi2_rule:.4f}, p = {p_chi2_rule:.4f}")

    print(f"\n  Panel (0,2) — Accuracy by trial type")
    for tt in ["match", "rule_order", "random"]:
        nc, nt, bt = tt_binom[tt]
        print(f"    {tt}: {nc}/{nt} = {nc/nt:.4f}, binomial p = {bt.pvalue:.4f}")
    print(f"    Chi-square (across trial types): chi2({dof_tt}) = {chi2_tt:.4f}, p = {p_chi2_tt:.4f}")
    for (a, b), p in tt_fisher.items():
        print(f"    Fisher exact ({a} vs {b}): p = {p:.4f}")

    print(f"\n  Panel (1,0) — RT distribution per participant (correct trials)")
    for sid, label in zip(sids, short_labels):
        print(f"    {label}: median RT = {rt_med_per_sid[sid]:.4f} s")

    print(f"\n  Panel (1,1) — RT by rule type (correct trials, aggregated)")
    print(f"    ABA: median = {np.median(rt_aba):.4f} s  (n={len(rt_aba)})")
    print(f"    ABB: median = {np.median(rt_abb):.4f} s  (n={len(rt_abb)})")
    print(f"    MWU (ABA vs ABB): U = {u_rule_rt:.1f}, p = {p_rule_rt:.4f}")

    print(f"\n  Panel (1,2) — RT by trial type (correct trials, aggregated)")
    for tt, rt_vals in [("match", rt_match), ("rule_order", rt_ro), ("random", rt_rand)]:
        print(f"    {tt}: median = {np.median(rt_vals):.4f} s  (n={len(rt_vals)})")
    print(f"    MWU (match vs rule_order):    U = {u_match_ro:.1f}, p = {p_match_ro:.4f}")
    print(f"    MWU (match vs random):        U = {u_match_rand:.1f}, p = {p_match_rand:.4f}")
    print(f"    MWU (rule_order vs random):   U = {u_ro_rand:.1f}, p = {p_ro_rand:.4f}")

    print("\n───────────────────────────────────────────────────────────────\n")

    # ── figure layout ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    # ── (0,0) Overall accuracy per participant ────────────────────────────────
    ax = axes[0, 0]
    bar_w = 0.4
    gap = 0.2
    x_pos = np.arange(n_subj) * (bar_w + gap)
    for i, sid in enumerate(sids):
        ax.bar(x_pos[i], overall_acc[sid], color=colors[i],
               label=short_labels[i], width=bar_w)
    ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels)
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
        chance=0.5,
    )

    # ── (0,2) Accuracy by trial type ─────────────────────────────────────────
    ax = axes[0, 2]
    trial_type_subject_vals = {
        sid: [trial_type_acc[sid]["match"],
              trial_type_acc[sid]["rule_order"],
              trial_type_acc[sid]["random"]]
        for sid in sids
    }
    touching_bars(
        ax=ax,
        groups=["Match", "Rule order", "Random"],
        subject_values=trial_type_subject_vals,
        colors=colors,
        ylabel="Accuracy",
        title="Accuracy by trial type",
        show_legend=False,
        ylim=(0.0, 1.05),
        chance=0.5,
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

    # ── (1,2) RT distribution by trial type (aggregated) ─────────────────────
    ax = axes[1, 2]
    for tt, color in TRIAL_TYPE_COLORS.items():
        rt_vals = df_correct.loc[df_correct["trial_type"] == tt, "rt"].dropna().to_numpy()
        label = tt.replace("_", " ")
        kde_with_median(ax, rt_vals, color, label)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT by trial type (aggregated)")
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
    parser = argparse.ArgumentParser(description="2×3 summary behavior figure for MEG task")
    parser.add_argument("--data-dir", required=True,
                        help="Root data directory containing date-based subfolders "
                             "with sub-*_events.csv files")
    parser.add_argument("--outdir", required=True,
                        help="Directory to write the figure")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_sessions(data_dir)

    if len(df) == 0:
        raise ValueError("No trial data found.")

    make_summary_figure(df, outdir / "fig_summary.png")


if __name__ == "__main__":
    main()
