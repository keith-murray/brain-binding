#!/usr/bin/env python3
"""
Behavioral analysis script for the ABA/ABB relational inference task.

What it does
------------
1. Loads all session CSVs under a root data directory
2. Produces summary tables
3. Runs the following analyses / plots:

Main task:
    - overall accuracy vs chance
    - accuracy by rule type (ABA vs ABB)
    - RT by rule type (correct trials)
    - learning curves across trials
    - RT distributions (correct vs incorrect)
    - error pattern analysis (A' / B' / other)
    - subject-level RT-accuracy relationship
    - sequential effects (rule repeat vs switch)

WM block:
    - accuracy by ISI
    - RT by ISI
    - optional subject-level WM summaries

Outputs
-------
Creates an output directory containing:
    - CSV summary tables
    - PNG figures

Usage
-----
python behavioral_analysis.py --data-root data --outdir behavior_results

Notes
-----
- This script assumes the directory / file structure described in DATA_FORMAT.md.
- It does not fit a DDM.
- It uses simple inferential statistics where possible and leaves more advanced mixed models optional.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# I/O
# -----------------------------

def find_csv_files(data_root: Path) -> list[Path]:
    csvs = sorted(data_root.glob("*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_root}")
    return csvs


def load_sessions(data_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in find_csv_files(data_root):
        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)
        df["session_folder"] = csv_path.parent.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).fillna(False)

    # Normalize dtypes
    str_cols = [
        "block", "sid", "sample_stim", "rule_type", "A_stim", "B_stim",
        "A_prime", "B_prime", "rule_sequence", "test_sequence",
        "correct_next_stim", "slot_mapping", "correct_stim", "response_stim"
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    numeric_cols = [
        "trial", "isi_condition", "response_key", "response_slot", "correct", "rt",
        "isi1", "isi2", "isi3", "isi4", "isi5", "iti",
        "t_fixation", "t_sample", "t_delay",
        "t_rule1", "t_rule2", "t_rule3", "t_test1", "t_test2", "t_response"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Harmonize correct as integer where possible
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype("Int64")

    return df


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sem(x: Sequence[float]) -> float:
    x = pd.Series(x).dropna().to_numpy()
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def chance_level_main(n_choices: int = 4) -> float:
    return 1.0 / n_choices


def add_main_trial_index(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Create a within-main-task trial index per subject: 1..120
    """
    out = df_main.copy()
    out = out.sort_values(["sid", "trial"]).reset_index(drop=True)
    out["main_trial_idx"] = out.groupby("sid").cumcount() + 1
    return out


def categorize_error_type(row: pd.Series) -> Optional[str]:
    """
    For incorrect main-task trials:
      - "A_prime"
      - "B_prime"
      - "other"
    Returns None for correct / missing responses.
    """
    if pd.isna(row.get("correct")) or int(row["correct"]) == 1:
        return None

    resp = row.get("response_stim")
    if pd.isna(resp):
        return "no_response"

    if resp == row.get("A_prime"):
        return "A_prime"
    if resp == row.get("B_prime"):
        return "B_prime"
    return "other"



def subject_summary_main(df_main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df_main.groupby("sid"):
        acc = g["correct"].mean()
        rt_correct = g.loc[g["correct"] == 1, "rt"].median()
        rt_all = g["rt"].median()
        n = len(g)
        rows.append({
            "sid": sid,
            "n_trials": n,
            "accuracy": acc,
            "median_rt_correct": rt_correct,
            "median_rt_all": rt_all,
        })
    return pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)


def subject_summary_wm(df_wm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df_wm.groupby("sid"):
        rows.append({
            "sid": sid,
            "n_trials": len(g),
            "accuracy": g["correct"].mean(),
            "median_rt_correct": g.loc[g["correct"] == 1, "rt"].median(),
            "median_rt_all": g["rt"].median(),
        })
    return pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)


def paired_t_with_effect(x: pd.Series, y: pd.Series) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 2:
        return {"n": len(d), "t": np.nan, "p": np.nan, "cohens_dz": np.nan}
    t, p = stats.ttest_rel(d.iloc[:, 0], d.iloc[:, 1])
    diff = d.iloc[:, 0] - d.iloc[:, 1]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    return {"n": len(d), "t": t, "p": p, "cohens_dz": dz}


# -----------------------------
# Plotting utilities
# -----------------------------

def savefig(path: Path, dpi: int = 180) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def box_with_points(
    subject_points: pd.DataFrame,
    ylabel: str,
    title: str,
    outpath: Path,
    chance: Optional[float] = None,
) -> None:
    cols = [c for c in subject_points.columns if subject_points[c].notna().any()]
    data = [subject_points[c].dropna().to_numpy() for c in cols]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, alpha=0.75, s=28, zorder=3)

    if chance is not None:
        ax.axhline(chance, linestyle="--", linewidth=1)

    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    savefig(outpath)


# -----------------------------
# Analyses
# -----------------------------

def analysis_overall_accuracy(df_main: pd.DataFrame, outdir: Path) -> None:
    subj = df_main.groupby("sid")["correct"].mean().rename("accuracy").reset_index()
    save_df(subj, outdir / "main_subject_overall_accuracy.csv")

    mean_acc = subj["accuracy"].mean()
    se_acc = sem(subj["accuracy"])
    chance = chance_level_main(4)

    # One-sample t-test vs chance at subject level
    if len(subj) >= 2:
        t, p = stats.ttest_1samp(subj["accuracy"], popmean=chance)
    else:
        t, p = np.nan, np.nan

    stats_df = pd.DataFrame([{
        "n_subjects": subj["sid"].nunique(),
        "mean_accuracy": mean_acc,
        "sem_accuracy": se_acc,
        "chance": chance,
        "t_vs_chance": t,
        "p_vs_chance": p,
    }])
    save_df(stats_df, outdir / "main_overall_accuracy_stats.csv")

    pts = pd.DataFrame({"Overall": subj["accuracy"].values})

    box_with_points(
        subject_points=pts,
        ylabel="Accuracy",
        title="Main task overall accuracy",
        outpath=outdir / "fig1_main_overall_accuracy.png",
        chance=chance,
    )


def analysis_rule_type(df_main: pd.DataFrame, outdir: Path) -> None:
    subj_acc = (
        df_main.groupby(["sid", "rule_type"])["correct"]
        .mean()
        .unstack("rule_type")
        .sort_index()
    )
    save_df(subj_acc.reset_index(), outdir / "main_subject_accuracy_by_rule_type.csv")

    acc_stats = paired_t_with_effect(subj_acc.get("ABA"), subj_acc.get("ABB"))
    save_df(pd.DataFrame([acc_stats]), outdir / "main_accuracy_by_rule_type_stats.csv")

    box_with_points(
        subject_points=subj_acc,
        ylabel="Accuracy",
        title="Accuracy by rule type",
        outpath=outdir / "fig1_accuracy_by_rule_type.png",
    )

    subj_rt = (
        df_main.loc[df_main["correct"] == 1]
        .groupby(["sid", "rule_type"])["rt"]
        .median()
        .unstack("rule_type")
        .sort_index()
    )
    save_df(subj_rt.reset_index(), outdir / "main_subject_rt_by_rule_type.csv")

    rt_stats = paired_t_with_effect(subj_rt.get("ABA"), subj_rt.get("ABB"))
    save_df(pd.DataFrame([rt_stats]), outdir / "main_rt_by_rule_type_stats.csv")

    box_with_points(
        subject_points=subj_rt,
        ylabel="Median RT (s), correct trials",
        title="Correct-trial RT by rule type",
        outpath=outdir / "fig1_rt_by_rule_type.png",
    )


def analysis_learning_curve(df_main: pd.DataFrame, outdir: Path, window: int = 10) -> None:
    df_main = add_main_trial_index(df_main)

    rows = []
    for sid, g in df_main.groupby("sid"):
        g = g.sort_values("main_trial_idx").copy()
        g["acc_roll"] = g["correct"].rolling(window=window, min_periods=1).mean()
        g["rt_roll"] = g["rt"].rolling(window=window, min_periods=1).median()
        rows.append(g[["sid", "main_trial_idx", "rule_type", "acc_roll", "rt_roll"]])

    roll = pd.concat(rows, ignore_index=True)
    save_df(roll, outdir / "main_learning_curve_long.csv")

    grp = roll.groupby("main_trial_idx")
    mean_acc = grp["acc_roll"].mean()
    sem_acc = grp["acc_roll"].apply(sem)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mean_acc.index, mean_acc.values, label="All trials")
    ax.fill_between(
        mean_acc.index,
        mean_acc.values - sem_acc.values,
        mean_acc.values + sem_acc.values,
        alpha=0.25,
    )
    ax.set_xlabel("Main-task trial index")
    ax.set_ylabel(f"Rolling accuracy (window={window})")
    ax.set_title("Learning curve")
    ax.set_ylim(0, 1)
    savefig(outdir / "fig1_learning_curve_accuracy.png")

    # Separate by rule type
    fig, ax = plt.subplots(figsize=(7, 4))
    for rule in ["ABA", "ABB"]:
        sub = roll.loc[roll["rule_type"] == rule]
        grp_r = sub.groupby("main_trial_idx")["acc_roll"]
        m = grp_r.mean()
        s = grp_r.apply(sem)
        ax.plot(m.index, m.values, label=rule)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values, alpha=0.2)
    ax.set_xlabel("Main-task trial index")
    ax.set_ylabel(f"Rolling accuracy (window={window})")
    ax.set_title("Learning curve by rule type")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    savefig(outdir / "fig1_learning_curve_accuracy_by_rule.png")

    # Simple subject-level early-vs-late comparison
    n_per_subject = df_main.groupby("sid").size().min()
    cut = n_per_subject // 2
    tmp = df_main.copy()
    tmp["half"] = tmp["main_trial_idx"].apply(lambda x: "early" if x <= cut else "late")
    subj_half = tmp.groupby(["sid", "half"])["correct"].mean().unstack("half")
    save_df(subj_half.reset_index(), outdir / "main_subject_accuracy_early_vs_late.csv")
    save_df(pd.DataFrame([paired_t_with_effect(subj_half["late"], subj_half["early"])]),
            outdir / "main_accuracy_early_vs_late_stats.csv")


def analysis_rt_correct_vs_incorrect(df_main: pd.DataFrame, outdir: Path) -> None:
    subj = (
        df_main.groupby(["sid", "correct"])["rt"]
        .median()
        .unstack("correct")
        .rename(columns={0: "incorrect", 1: "correct"})
    )
    save_df(subj.reset_index(), outdir / "main_subject_rt_correct_vs_incorrect.csv")

    stats_df = pd.DataFrame([paired_t_with_effect(subj.get("correct"), subj.get("incorrect"))])
    save_df(stats_df, outdir / "main_rt_correct_vs_incorrect_stats.csv")

    box_with_points(
        subject_points=subj,
        ylabel="Median RT (s)",
        title="RT by response correctness",
        outpath=outdir / "fig1_rt_correct_vs_incorrect.png",
    )

    # RT distributions pooled across subjects
    fig, ax = plt.subplots(figsize=(7, 4))
    for corr, label in [(1, "correct"), (0, "incorrect")]:
        vals = df_main.loc[df_main["correct"] == corr, "rt"].dropna().to_numpy()
        if len(vals) > 0:
            ax.hist(vals, bins=25, alpha=0.5, density=True, label=label)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT distribution")
    ax.legend(frameon=False)
    savefig(outdir / "fig1_rt_distribution.png")


def analysis_error_patterns(df_main: pd.DataFrame, outdir: Path) -> None:
    tmp = df_main.copy()
    tmp["error_type"] = tmp.apply(categorize_error_type, axis=1)
    err = tmp.loc[tmp["correct"] == 0].copy()

    if len(err) == 0:
        save_df(pd.DataFrame(), outdir / "main_error_pattern_subject_proportions.csv")
        return

    subj_counts = (
        err.groupby(["sid", "error_type"])
        .size()
        .rename("n")
        .reset_index()
    )
    subj_totals = subj_counts.groupby("sid")["n"].sum().rename("total_errors").reset_index()
    subj_counts = subj_counts.merge(subj_totals, on="sid", how="left")
    subj_counts["prop"] = subj_counts["n"] / subj_counts["total_errors"]

    pivot = subj_counts.pivot(index="sid", columns="error_type", values="prop").fillna(0.0)
    save_df(pivot.reset_index(), outdir / "main_error_pattern_subject_proportions.csv")

    box_with_points(
        subject_points=pivot,
        ylabel="Proportion of errors",
        title="Error pattern analysis",
        outpath=outdir / "fig1_error_patterns.png",
    )



def analysis_rt_accuracy_tradeoff(df_main: pd.DataFrame, outdir: Path) -> None:
    subj = subject_summary_main(df_main)
    save_df(subj, outdir / "main_subject_summary.csv")

    x = subj["median_rt_correct"]
    y = subj["accuracy"]
    valid = ~(x.isna() | y.isna())
    if valid.sum() >= 2:
        r, p = stats.pearsonr(x[valid], y[valid])
    else:
        r, p = np.nan, np.nan

    save_df(pd.DataFrame([{
        "n_subjects": int(valid.sum()),
        "pearson_r": r,
        "p_value": p,
    }]), outdir / "main_rt_accuracy_tradeoff_stats.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=40)
    for _, row in subj.iterrows():
        ax.annotate(str(row["sid"])[:6], (row["median_rt_correct"], row["accuracy"]),
                    fontsize=8, alpha=0.8)
    ax.set_xlabel("Median RT (s), correct trials")
    ax.set_ylabel("Accuracy")
    ax.set_title("Subject-level RT–accuracy relationship")
    savefig(outdir / "fig1_subject_rt_accuracy_tradeoff.png")


def analysis_sequential_effects(df_main: pd.DataFrame, outdir: Path) -> None:
    tmp = df_main.sort_values(["sid", "trial"]).copy()
    tmp["prev_rule_type"] = tmp.groupby("sid")["rule_type"].shift(1)
    tmp["transition"] = np.where(
        tmp["prev_rule_type"].isna(),
        "first",
        np.where((tmp["rule_type"] == tmp["prev_rule_type"]).fillna(False), "repeat", "switch")
    )

    tmp_valid = tmp.loc[tmp["transition"].isin(["repeat", "switch"])].copy()

    subj = (
        tmp_valid.groupby(["sid", "transition"])["correct"]
        .mean()
        .unstack("transition")
    )
    save_df(subj.reset_index(), outdir / "main_subject_accuracy_by_transition.csv")
    save_df(pd.DataFrame([paired_t_with_effect(subj.get("repeat"), subj.get("switch"))]),
            outdir / "main_accuracy_by_transition_stats.csv")

    box_with_points(
        subject_points=subj,
        ylabel="Accuracy",
        title="Sequential effect: rule repeat vs switch",
        outpath=outdir / "fig1_accuracy_repeat_vs_switch.png",
    )

    subj_rt = (
        tmp_valid.loc[tmp_valid["correct"] == 1]
        .groupby(["sid", "transition"])["rt"]
        .median()
        .unstack("transition")
    )
    save_df(subj_rt.reset_index(), outdir / "main_subject_rt_by_transition.csv")
    save_df(pd.DataFrame([paired_t_with_effect(subj_rt.get("repeat"), subj_rt.get("switch"))]),
            outdir / "main_rt_by_transition_stats.csv")

    box_with_points(
        subject_points=subj_rt,
        ylabel="Median RT (s), correct trials",
        title="Sequential RT effect: rule repeat vs switch",
        outpath=outdir / "fig1_rt_repeat_vs_switch.png",
    )


def analysis_stimulus_identity_controls(df_main: pd.DataFrame, outdir: Path) -> None:
    """
    Simple descriptive control analysis:
    accuracy by A_stim, B_stim, A_prime, B_prime.
    This is not a full crossed model, but it is useful as a figure supplement
    or sanity check for token-invariance.
    """
    for col in ["A_stim", "B_stim", "A_prime", "B_prime"]:
        subj = (
            df_main.groupby(["sid", col])["correct"]
            .mean()
            .unstack(col)
            .sort_index(axis=1)
        )
        save_df(subj.reset_index(), outdir / f"main_subject_accuracy_by_{col}.csv")

        box_with_points(
            subject_points=subj,
            ylabel="Accuracy",
            title=f"Accuracy by {col}",
            outpath=outdir / f"fig_sup_accuracy_by_{col}.png",
        )


def analysis_per_subject(df_main: pd.DataFrame, outdir: Path, window: int = 10) -> None:
    subj_dir = outdir / "per_subject"
    ensure_dir(subj_dir)

    df_main = add_main_trial_index(df_main)

    for sid, g in df_main.groupby("sid"):
        g = g.sort_values("main_trial_idx").copy()
        g["acc_roll"] = g["correct"].rolling(window=window, min_periods=1).mean()

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(f"Subject {sid}", fontsize=12)

        # Panel 1: learning curve
        ax = axes[0, 0]
        for rule, color in [("ABA", "C0"), ("ABB", "C1")]:
            sub = g.loc[g["rule_type"] == rule]
            if len(sub) > 0:
                roll = sub["correct"].rolling(window=window, min_periods=1).mean()
                ax.plot(sub["main_trial_idx"], roll, label=rule, color=color, alpha=0.8)
        ax.axhline(chance_level_main(4), linestyle="--", linewidth=1, color="gray")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel(f"Rolling accuracy (w={window})")
        ax.set_title("Learning curve")
        ax.legend(frameon=False, fontsize=8)

        # Panel 2: accuracy by rule type
        ax = axes[0, 1]
        acc_by_rule = g.groupby("rule_type")["correct"].mean()
        colors = ["C0" if r == "ABA" else "C1" for r in acc_by_rule.index]
        ax.bar(acc_by_rule.index, acc_by_rule.values, color=colors, alpha=0.7)
        ax.axhline(chance_level_main(4), linestyle="--", linewidth=1, color="gray")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by rule type")

        # Panel 3: RT distribution (correct vs incorrect)
        ax = axes[1, 0]
        for corr, label, color in [(1, "correct", "C2"), (0, "incorrect", "C3")]:
            vals = g.loc[g["correct"] == corr, "rt"].dropna().to_numpy()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.5, density=True, label=label, color=color)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
        ax.set_title("RT distribution")
        ax.legend(frameon=False, fontsize=8)

        # Panel 4: median RT by rule type (correct trials only)
        ax = axes[1, 1]
        rt_by_rule = g.loc[g["correct"] == 1].groupby("rule_type")["rt"].median()
        colors = ["C0" if r == "ABA" else "C1" for r in rt_by_rule.index]
        ax.bar(rt_by_rule.index, rt_by_rule.values, color=colors, alpha=0.7)
        ax.set_ylabel("Median RT (s), correct trials")
        ax.set_title("RT by rule type")

        savefig(subj_dir / f"subject_{sid}.png")


def analysis_wm_block(df_wm: pd.DataFrame, outdir: Path) -> None:
    subj = subject_summary_wm(df_wm)
    save_df(subj, outdir / "wm_subject_summary.csv")

    # Accuracy by ISI
    subj_acc = (
        df_wm.groupby(["sid", "isi_condition"])["correct"]
        .mean()
        .unstack("isi_condition")
        .sort_index(axis=1)
    )
    save_df(subj_acc.reset_index(), outdir / "wm_subject_accuracy_by_isi.csv")

    box_with_points(
        subject_points=subj_acc.rename(columns=lambda x: f"{x:.1f}s"),
        ylabel="Accuracy",
        title="WM localizer accuracy by delay",
        outpath=outdir / "fig1_wm_accuracy_by_isi.png",
        chance=chance_level_main(4),
    )

    if set(subj_acc.columns.tolist()) >= {1.0, 2.5}:
        stat_acc = paired_t_with_effect(subj_acc[1.0], subj_acc[2.5])
        save_df(pd.DataFrame([stat_acc]), outdir / "wm_accuracy_by_isi_stats.csv")

    # RT by ISI (correct trials)
    subj_rt = (
        df_wm.loc[df_wm["correct"] == 1]
        .groupby(["sid", "isi_condition"])["rt"]
        .median()
        .unstack("isi_condition")
        .sort_index(axis=1)
    )
    save_df(subj_rt.reset_index(), outdir / "wm_subject_rt_by_isi.csv")

    box_with_points(
        subject_points=subj_rt.rename(columns=lambda x: f"{x:.1f}s"),
        ylabel="Median RT (s), correct trials",
        title="WM localizer RT by delay",
        outpath=outdir / "fig1_wm_rt_by_isi.png",
    )

    if set(subj_rt.columns.tolist()) >= {1.0, 2.5}:
        stat_rt = paired_t_with_effect(subj_rt[1.0], subj_rt[2.5])
        save_df(pd.DataFrame([stat_rt]), outdir / "wm_rt_by_isi_stats.csv")


def write_overall_counts(df: pd.DataFrame, outdir: Path) -> None:
    counts = pd.DataFrame([{
        "n_rows_total": len(df),
        "n_subjects": df["sid"].nunique(),
        "n_wm_trials": int((df["block"] == "WM").sum()),
        "n_main_trials": int(df["block"].astype(str).str.startswith("main_run").sum()),
        "n_missing_rt_total": int(df["rt"].isna().sum()),
    }])
    save_df(counts, outdir / "dataset_counts.csv")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to root data directory containing session subfolders.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory to write figures and summary tables.")
    parser.add_argument("--learning-window", type=int, default=10,
                        help="Rolling window size for learning-curve plots.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_sessions(data_root)
    write_overall_counts(df, outdir)

    df_main = df.loc[df["block"].astype(str).str.startswith("main_run")].copy()
    df_wm = df.loc[df["block"] == "WM"].copy()

    if len(df_main) == 0:
        raise ValueError("No main-task trials found.")
    if len(df_wm) == 0:
        print("Warning: no WM trials found. Skipping WM analyses.")

    # Main task analyses
    analysis_overall_accuracy(df_main, outdir)
    analysis_rule_type(df_main, outdir)
    analysis_learning_curve(df_main, outdir, window=args.learning_window)
    analysis_rt_correct_vs_incorrect(df_main, outdir)
    analysis_error_patterns(df_main, outdir)
    analysis_rt_accuracy_tradeoff(df_main, outdir)
    analysis_sequential_effects(df_main, outdir)
    analysis_stimulus_identity_controls(df_main, outdir)
    analysis_per_subject(df_main, outdir, window=args.learning_window)

    # WM analyses
    if len(df_wm) > 0:
        analysis_wm_block(df_wm, outdir)

    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
