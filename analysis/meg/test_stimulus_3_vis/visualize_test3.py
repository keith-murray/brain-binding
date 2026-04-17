#!/usr/bin/env python3
"""
Visualize MEG magnetometer time courses locked to the 3rd test stimulus.

The 3rd test stimulus is the key event: it reveals whether the trial is a
match, rule-order mismatch, or random mismatch — and we saw behaviorally
that random mismatches elicit faster RTs.  This script plots the evoked
magnetometer response for all three conditions.

The epochs file records only two trigger codes for the 3rd test stimulus:
  test_match     (event_id 7) — 60 trials
  test_non_match (event_id 8) — 60 trials

To split test_non_match into rule_order (n=30) and random (n=30) we
cross-reference the behavioral CSV: the trial order in the epochs
matches the trial order in the CSV, so we do a sequential 1-to-1 mapping.

Output
------
analysis/meg/figures/fig_test3_evoked.png

Usage
-----
uv run python analysis/meg/visualize_test3.py \
    --epochs  data/sub-001_ses-01_task-binding_proc-clean_epo.fif \
    --beh     data/2026-04-10/sub-001_events.csv \
    --outdir  analysis/meg/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

# Condition display settings
CONDITIONS = {
    "match":      {"color": "#4477AA", "label": "Match"},
    "rule_order": {"color": "#EE6677", "label": "Rule order"},
    "random":     {"color": "#228833", "label": "Random"},
}


# =============================================================================
# Data loading & cross-referencing
# =============================================================================

def load_and_label(epochs_path: str, beh_path: str) -> dict[str, mne.Epochs]:
    """Load epochs and split test_non_match into rule_order / random.

    Returns a dict mapping condition name → Epochs object.
    """
    epo = mne.read_epochs(epochs_path, preload=True)
    df = pd.read_csv(beh_path).sort_values("trial").reset_index(drop=True)

    # ── 1. Isolate the 3rd-test-stimulus rows from metadata ──────────────────
    meta = epo.metadata.copy()
    test_mask = meta["event_name"].isin(["test_match", "test_non_match"])
    meta_test = meta[test_mask]  # 120 rows, in epoch order

    # ── 2. Verify sequences agree ─────────────────────────────────────────────
    epo_seq = (meta_test["event_name"] == "test_match").astype(int).values
    beh_seq = df["match"].values
    if not np.array_equal(epo_seq, beh_seq):
        raise ValueError(
            "Match/non-match sequence in epochs does not match behavioral CSV. "
            "Cannot safely cross-reference by trial order."
        )

    # ── 3. Propagate mismatch_type into metadata ──────────────────────────────
    meta["mismatch_type"] = ""
    meta.loc[meta_test.index, "mismatch_type"] = df["mismatch_type"].fillna("").values
    epo.metadata = meta

    # ── 4. Build per-condition epochs ─────────────────────────────────────────
    epo_match = epo["test_match"]
    epo_ro    = epo[epo.metadata["mismatch_type"] == "rule_order"]
    epo_rand  = epo[epo.metadata["mismatch_type"] == "random"]

    print(
        f"Epoch counts — match: {len(epo_match)}, "
        f"rule_order: {len(epo_ro)}, random: {len(epo_rand)}"
    )

    return {"match": epo_match, "rule_order": epo_ro, "random": epo_rand}


# =============================================================================
# Figure
# =============================================================================

def plot_evoked(condition_epochs: dict[str, mne.Epochs], outpath: Path) -> None:
    """2-row figure:
      Row 0 — butterfly plots (all magnetometers) for each condition
      Row 1 — GFP comparison across conditions on a single axis
    """
    evokeds = {k: v.average() for k, v in condition_epochs.items()}

    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    # ── Row 0: butterfly plots ────────────────────────────────────────────────
    for col, (cond, cfg) in enumerate(CONDITIONS.items()):
        ax = fig.add_subplot(2, 3, col + 1)
        evk = evokeds[cond]
        times = evk.times
        data = evk.copy().pick("mag").get_data()   # (n_mag, n_times)
        scale = 1e15                                # T → fT

        # individual channel traces
        ax.plot(times, data.T * scale,
                color=cfg["color"], alpha=0.25, linewidth=0.4)

        # GFP (bold line)
        gfp = np.sqrt(np.mean(data ** 2, axis=0))
        ax.plot(times, gfp * scale,
                color=cfg["color"], linewidth=2.0, label="GFP")

        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=3)
        ax.axhline(0, color="gray",  linewidth=0.5, linestyle="-",  zorder=0)

        n = len(condition_epochs[cond])
        ax.set_title(f"{cfg['label']}  (n={n})", fontsize=11)
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("fT")

    # ── Row 1: GFP comparison ─────────────────────────────────────────────────
    ax_gfp = fig.add_subplot(2, 1, 2)
    for cond, cfg in CONDITIONS.items():
        evk = evokeds[cond]
        data = evk.copy().pick("mag").get_data()
        gfp = np.sqrt(np.mean(data ** 2, axis=0))
        ax_gfp.plot(evk.times, gfp * 1e15,
                    color=cfg["color"], linewidth=2.0, label=cfg["label"])

    ax_gfp.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=3)
    ax_gfp.axhline(0, color="gray",  linewidth=0.5, linestyle="-",  zorder=0)
    ax_gfp.set_xlabel("Time (s)")
    ax_gfp.set_ylabel("GFP (fT)")
    ax_gfp.set_title("Global field power — all conditions")
    ax_gfp.legend(frameon=False, fontsize=9)

    # panel letters
    for ax, letter in zip(fig.axes[:4], "ABCD"):
        ax.text(-0.18, 1.07, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {outpath}")


# =============================================================================
# Topomap figure
# =============================================================================

TOPOMAP_TIMES = [0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350]  # seconds


def plot_topomaps(condition_epochs: dict[str, mne.Epochs], outpath: Path) -> None:
    """Grid of topomaps: rows = conditions, columns = time points.

    Each cell shows the magnetometer field map at that latency.
    A shared colour scale (vmin/vmax) is computed across all conditions and
    time points so maps are directly comparable.
    """
    # Pick only the radial (Z) component — one channel per sensor location,
    # required because MNE's topomap rejects overlapping positions.
    z_picks = [ch for ch in condition_epochs["match"].ch_names
               if ch.endswith(" Z") and
               mne.channel_type(condition_epochs["match"].info,
                                condition_epochs["match"].ch_names.index(ch)) == "mag"]
    evokeds = {k: v.average().pick(z_picks) for k, v in condition_epochs.items()}

    # ── shared colour scale ───────────────────────────────────────────────────
    all_vals = np.concatenate([
        evk.get_data()[:, evk.time_as_index(t)[0]]
        for evk in evokeds.values()
        for t in TOPOMAP_TIMES
    ])
    vmax = np.percentile(np.abs(all_vals), 98) * 1e15   # fT
    vmin = -vmax

    n_times = len(TOPOMAP_TIMES)
    n_conds = len(CONDITIONS)

    fig, axes = plt.subplots(
        n_conds, n_times,
        figsize=(2.0 * n_times, 2.4 * n_conds),
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row, (cond, cfg) in enumerate(CONDITIONS.items()):
        evk = evokeds[cond]

        # condition label on the left
        axes[row, 0].set_ylabel(cfg["label"], fontsize=11, labelpad=4)

        for col, t in enumerate(TOPOMAP_TIMES):
            ax = axes[row, col]
            t_idx = evk.time_as_index(t)[0]
            data_fT = evk.get_data()[:, t_idx] * 1e15

            im, _ = mne.viz.plot_topomap(
                data_fT,
                evk.info,
                axes=ax,
                show=False,
                vlim=(vmin, vmax),
                cmap="RdBu_r",
                contours=4,
                sensors=False,
            )

            if row == 0:
                ax.set_title(f"{int(t * 1000)} ms", fontsize=10)

    # shared colorbar on the right
    cbar = fig.colorbar(
        im,
        ax=axes[:, -1].tolist(),
        orientation="vertical",
        fraction=0.04,
        pad=0.02,
        shrink=0.8,
    )
    cbar.set_label("fT", fontsize=10)

    fig.suptitle("Magnetometer topographies — 3rd test stimulus onset",
                 fontsize=12, y=1.01)

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {outpath}")


# =============================================================================
# Combined GFP + 200 ms topomap figure
# =============================================================================

def plot_gfp_with_topomap(condition_epochs: dict[str, mne.Epochs],
                           outpath: Path,
                           topo_time: float = 0.200) -> None:
    """GFP comparison (top) + topomaps at a single time point (bottom).

    A vertical dashed line on the GFP panel marks topo_time, visually
    linking it to the maps below.
    """
    z_picks = [ch for ch in condition_epochs["match"].ch_names
               if ch.endswith(" Z") and
               mne.channel_type(condition_epochs["match"].info,
                                condition_epochs["match"].ch_names.index(ch)) == "mag"]

    evokeds_full = {k: v.average() for k, v in condition_epochs.items()}
    evokeds_z    = {k: v.pick(z_picks) for k, v in evokeds_full.items()}

    # ── shared colour scale for topomaps ─────────────────────────────────────
    all_topo = np.concatenate([
        evk.get_data()[:, evk.time_as_index(topo_time)[0]]
        for evk in evokeds_z.values()
    ])
    vmax = np.percentile(np.abs(all_topo), 98) * 1e15
    vmin = -vmax

    # ── layout: 2 rows, 3 cols; GFP spans all 3 cols in row 0 ────────────────
    fig = plt.figure(figsize=(10, 7))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.15,
                            height_ratios=[1.4, 1])

    # ── Row 0: GFP ────────────────────────────────────────────────────────────
    ax_gfp = fig.add_subplot(gs[0, :])
    for cond, cfg in CONDITIONS.items():
        evk  = evokeds_full[cond]
        data = evk.copy().pick("mag").get_data()
        gfp  = np.sqrt(np.mean(data ** 2, axis=0))
        ax_gfp.plot(evk.times, gfp * 1e15,
                    color=cfg["color"], linewidth=2.0, label=cfg["label"])

    ax_gfp.axvline(0,         color="black", linewidth=1.0, linestyle="--", zorder=3)
    ax_gfp.axvline(topo_time, color="black", linewidth=1.5, linestyle=":",  zorder=3)
    ax_gfp.axhline(0,         color="gray",  linewidth=0.5, linestyle="-",  zorder=0)
    ax_gfp.set_xlabel("Time (s)")
    ax_gfp.set_ylabel("GFP (fT)")
    ax_gfp.set_title("Global field power — test stimulus 3, all conditions")
    ax_gfp.legend(frameon=False, fontsize=9)
    ax_gfp.text(topo_time + 0.005, ax_gfp.get_ylim()[1] * 0.97,
                f"{int(topo_time * 1000)} ms", fontsize=9, va="top")

    # ── Row 1: topomaps at topo_time ─────────────────────────────────────────
    for col, (cond, cfg) in enumerate(CONDITIONS.items()):
        ax = fig.add_subplot(gs[1, col])
        evk    = evokeds_z[cond]
        t_idx  = evk.time_as_index(topo_time)[0]
        data_fT = evk.get_data()[:, t_idx] * 1e15

        im, _ = mne.viz.plot_topomap(
            data_fT, evk.info,
            axes=ax, show=False,
            vlim=(vmin, vmax),
            cmap="RdBu_r",
            contours=4,
            sensors=False,
        )
        ax.set_title(cfg["label"], fontsize=11)

    # shared colorbar
    cbar = fig.colorbar(im, ax=fig.axes[1:], orientation="vertical",
                        fraction=0.03, pad=0.04, shrink=0.85)
    cbar.set_label("fT", fontsize=10)

    # panel letters
    for ax, letter in zip([ax_gfp] + fig.axes[1:4], "ABCD"):
        ax.text(-0.10, 1.07, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {outpath}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MEG magnetometers locked to 3rd test stimulus"
    )
    parser.add_argument("--epochs", required=True,
                        help="Path to the .fif epochs file")
    parser.add_argument("--beh", required=True,
                        help="Path to sub-*_events.csv behavioral file")
    parser.add_argument("--outdir", required=True,
                        help="Directory to write the figure")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    condition_epochs = load_and_label(args.epochs, args.beh)
    plot_evoked(condition_epochs, outdir / "fig_test3_evoked.png")
    plot_topomaps(condition_epochs, outdir / "fig_test3_topomap.png")
    plot_gfp_with_topomap(condition_epochs, outdir / "fig_test3_gfp_topo200.png")


if __name__ == "__main__":
    main()
