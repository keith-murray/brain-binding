"""
rsa_analysis.py
---------------
RSA analysis + visualization, downstream of LSS first-level GLM.

Changes vs previous version
----------------------------
1. Cleaner RDM tick labels  (3-line: shape / role-symbol / phase+rule)
2. rule_type_test model RDM added to test-phase set
3. Brain maps produced in 3 variants:
     full    -- symmetric colorbar (pos + neg)
     pos     -- positive values only
     top10   -- top 10% of positive values only

Expects:
    {results_dir}/{subject}_lss_beta_4d.nii.gz
    {results_dir}/{subject}_lss_conditions.csv
"""

from __future__ import annotations

import argparse
import os
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONDITION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def parse_condition(label: str) -> dict:
    parts = str(label).split("_")
    shape, role, phase = parts[0], parts[1], parts[2]
    rule_type = parts[3] if len(parts) == 4 else None
    return {
        "shape":     shape,
        "role":      role,
        "phase":     phase,
        "rule_type": rule_type,
        "epoch":     "rule" if phase.startswith("rule") else "test",
    }


# ── Clean label symbols ───────────────────────────────────────────────────────
_ROLE_SYMBOL = {
    "A":      "A",
    "B":      "B",
    "Aprime": "A'",
    "Bprime": "B'",
}
_PHASE_SHORT = {
    "rule1": "r1",
    "rule2": "r2",
    "rule3": "r3",
    "test1": "t1",
    "test2": "t2",
}
_SHAPE_SHORT = {
    "circle":    "Cir",
    "rectangle": "Rec",
    "star":      "Sta",
    "triangle":  "Tri",
}


def short_label(cond: str) -> str:
    """
    3-line tick label:
      Line 1 — shape abbrev   (Cir / Rec / Sta / Tri)
      Line 2 — role symbol    (A / B / A' / B')
      Line 3 — phase + rule   (r1 / t1_ABA / etc.)
    """
    p     = parse_condition(cond)
    shape = _SHAPE_SHORT.get(p["shape"], p["shape"][:3].capitalize())
    role  = _ROLE_SYMBOL.get(p["role"],  p["role"])
    phase = _PHASE_SHORT.get(p["phase"], p["phase"])
    rule  = f"_{p['rule_type']}" if p["rule_type"] else ""
    return f"{shape}\n{role}\n{phase}{rule}"


def filter_test(conditions: list[str]) -> list[str]:
    return [c for c in conditions
            if parse_condition(c)["phase"].startswith("test")]


def abstract_role(role: str) -> str:
    return "A" if role in {"A", "Aprime"} else "B"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD CACHED LSS BETAS
# ══════════════════════════════════════════════════════════════════════════════

def load_lss_betas(results_dir: str, subject: str):
    beta_path   = os.path.join(results_dir, f"{subject}_lss_beta_4d.nii.gz")
    labels_path = os.path.join(results_dir, f"{subject}_lss_conditions.csv")
    if not os.path.exists(beta_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"LSS outputs not found for {subject} in {results_dir}.\n"
            f"  Expected: {beta_path}\n  and:      {labels_path}"
        )
    beta_4d    = nib.load(beta_path)
    conditions = pd.read_csv(labels_path).iloc[:, 0].tolist()
    return beta_4d, conditions


def subset_to_test(beta_4d, conditions: list[str]):
    from nilearn.image import index_img
    test_conds   = filter_test(conditions)
    test_indices = [conditions.index(c) for c in test_conds]
    return index_img(beta_4d, test_indices), test_conds


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL RDMs
# ══════════════════════════════════════════════════════════════════════════════

def _make_rdm(conditions: list[str], meta: pd.DataFrame,
              sim_fn) -> np.ndarray:
    n   = len(conditions)
    rdm = np.zeros((n, n))
    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            rdm[i, j] = 1.0 - float(sim_fn(meta.loc[ci], meta.loc[cj]))
    np.fill_diagonal(rdm, 0.0)
    return rdm


def make_model_rdms_full(conditions: list[str]) -> dict[str, np.ndarray]:
    meta = pd.DataFrame([parse_condition(c) for c in conditions],
                        index=conditions)
    return {
        "M1_shape": _make_rdm(
            conditions, meta,
            lambda a, b: float(a["shape"] == b["shape"])),
        "M2_abstract_role": _make_rdm(
            conditions, meta,
            lambda a, b: float(abstract_role(a["role"]) ==
                               abstract_role(b["role"]))),
        "M3_concrete_role": _make_rdm(
            conditions, meta,
            lambda a, b: float(a["role"] == b["role"])),
        "M4_epoch": _make_rdm(
            conditions, meta,
            lambda a, b: float(a["epoch"] == b["epoch"])),
        "M5_rule_type": _make_rdm(
            conditions, meta,
            lambda a, b: (0.5 if a["rule_type"] is None or
                          b["rule_type"] is None
                          else float(a["rule_type"] == b["rule_type"]))),
        "M6_binding": _make_rdm(
            conditions, meta,
            lambda a, b: float(
                a["shape"] == b["shape"] and
                abstract_role(a["role"]) == abstract_role(b["role"]))),
    }


def make_model_rdms_test(conditions: list[str]) -> dict[str, np.ndarray]:
    test_conds = filter_test(conditions)
    meta       = pd.DataFrame([parse_condition(c) for c in test_conds],
                               index=test_conds)

    def is_repeater(row):
        return (row["role"] == "Aprime" if row["rule_type"] == "ABA"
                else row["role"] == "Bprime")

    return {
        "shape_test": _make_rdm(
            test_conds, meta,
            lambda a, b: float(a["shape"] == b["shape"])),
        "abstract_test": _make_rdm(
            test_conds, meta,
            lambda a, b: float(a["role"] == b["role"])),
        "functional_test": _make_rdm(
            test_conds, meta,
            lambda a, b: float(is_repeater(a) == is_repeater(b))),
        # Rule-type model: test stimuli from same rule type (ABA vs ABB) cluster.
        # Analogous to M5_rule_type in the full set but restricted to test phase.
        # A rule-selective region shows high rho here — patterns cluster by which
        # abstract rule was active on that trial, orthogonal to stimulus identity.
        "rule_type_test": _make_rdm(
            test_conds, meta,
            lambda a, b: float(a["rule_type"] == b["rule_type"])),
    }


def rdm_to_vec(rdm: np.ndarray) -> np.ndarray:
    return squareform(rdm) if rdm.ndim == 2 else rdm


def save_model_rdms(results_dir: str, tag: str,
                    conditions: list[str],
                    model_rdms: dict[str, np.ndarray]) -> str:
    path = os.path.join(results_dir, f"model_rdms_{tag}.npz")
    np.savez(path, conditions=np.array(conditions),
             **{k: v for k, v in model_rdms.items()})
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SEARCHLIGHT CORE
# ══════════════════════════════════════════════════════════════════════════════

def _sphere_neural_vec(center_idx, beta_data, vox_coords,
                        vox_size_mm, radius_mm, n_cond):
    cv   = vox_coords[center_idx]
    d    = np.sqrt(((vox_coords - cv)**2 * vox_size_mm**2).sum(1))
    nb   = np.where(d <= radius_mm)[0]
    if len(nb) < n_cond:
        return None
    nv       = vox_coords[nb]
    patterns = beta_data[nv[:, 0], nv[:, 1], nv[:, 2], :].T
    return pdist(patterns, metric="correlation")


def _spearman_voxel(center_idx, beta_data, vox_coords, vox_size_mm,
                    model_vecs, model_names, n_cond, radius_mm):
    try:
        vec = _sphere_neural_vec(center_idx, beta_data, vox_coords,
                                  vox_size_mm, radius_mm, n_cond)
        if vec is None:
            return np.full(len(model_names), np.nan)
        return np.array([spearmanr(vec, model_vecs[n])[0]
                         for n in model_names], dtype=float)
    except Exception:
        return np.full(len(model_names), np.nan)


def _ols_voxel(center_idx, beta_data, vox_coords, vox_size_mm,
               model_vecs, model_names, n_cond, radius_mm, use_ranks):
    from numpy.linalg import lstsq
    try:
        vec = _sphere_neural_vec(center_idx, beta_data, vox_coords,
                                  vox_size_mm, radius_mm, n_cond)
        if vec is None:
            return np.full(len(model_names), np.nan)
        y = rankdata(vec).reshape(-1, 1) if use_ranks else vec.reshape(-1, 1)
        X = np.column_stack([
            model_vecs[n].astype(float) - model_vecs[n].mean()
            for n in model_names
        ])
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        betas, _, _, _ = lstsq(X, y, rcond=None)
        return np.asarray(betas[:-1, 0], dtype=float)
    except Exception:
        return np.full(len(model_names), np.nan)


def _sanitize(res_list, n_models):
    out = []
    for r in res_list:
        r = np.asarray(r, dtype=float)
        out.append(r if r.shape == (n_models,) else np.full(n_models, np.nan))
    return np.array(out)


def prepare_searchlight(beta_4d):
    data     = beta_4d.get_fdata()
    affine   = beta_4d.affine
    vox_size = np.abs(np.diag(affine)[:3])
    mask     = np.all(data != 0, axis=-1)
    coords   = np.array(np.where(mask)).T
    return data, affine, vox_size, coords, data.shape[:3]


def run_spearman_searchlight(beta_4d, model_rdms: dict,
                              radius_mm: float = 12.0, n_jobs: int = -1):
    from joblib import Parallel, delayed
    data, affine, vsz, coords, vshape = prepare_searchlight(beta_4d)
    n_cond = data.shape[-1]
    names  = list(model_rdms.keys())
    vecs   = {n: rdm_to_vec(rdm) for n, rdm in model_rdms.items()}
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_spearman_voxel)(
            i, data, coords, vsz, vecs, names, n_cond, radius_mm)
        for i in range(len(coords))
    )
    return _sanitize(res, len(names)), names, coords, affine, vshape


def run_ols_searchlight(beta_4d, model_rdms: dict,
                         radius_mm: float = 12.0,
                         use_ranks: bool = False, n_jobs: int = -1):
    from joblib import Parallel, delayed
    data, affine, vsz, coords, vshape = prepare_searchlight(beta_4d)
    n_cond = data.shape[-1]
    names  = list(model_rdms.keys())
    vecs   = {n: rdm_to_vec(rdm) for n, rdm in model_rdms.items()}
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_ols_voxel)(
            i, data, coords, vsz, vecs, names, n_cond, radius_mm, use_ranks)
        for i in range(len(coords))
    )
    return _sanitize(res, len(names)), names, coords, affine, vshape


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE / CACHE SEARCHLIGHT NIfTIs
# ══════════════════════════════════════════════════════════════════════════════

def save_maps(results_dir, subject, prefix, results_arr,
              model_names, vox_coords, affine, vol_shape) -> dict[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    paths = {}
    for m_idx, name in enumerate(model_names):
        vol = np.full(vol_shape, np.nan, dtype=float)
        vol[vox_coords[:, 0],
            vox_coords[:, 1],
            vox_coords[:, 2]] = results_arr[:, m_idx]
        img  = nib.Nifti1Image(vol, affine)
        path = os.path.join(results_dir, f"{subject}_{prefix}_{name}.nii.gz")
        img.to_filename(path)
        paths[name] = path
        print(f"    saved {path}  "
              f"mean={np.nanmean(vol):.4f}  max={np.nanmax(vol):.4f}")
    return paths


def maps_cached(results_dir, subject, prefix, model_names) -> bool:
    return all(
        os.path.exists(
            os.path.join(results_dir, f"{subject}_{prefix}_{n}.nii.gz"))
        for n in model_names
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COORDINATE UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def mni_to_vox(mni_coord, affine) -> np.ndarray:
    inv = np.linalg.inv(affine)
    vox = inv[:3, :3] @ np.array(mni_coord) + inv[:3, 3]
    return np.round(vox).astype(int)


def extract_sphere(beta_4d, center_vox, radius_mm):
    data     = beta_4d.get_fdata()
    affine   = beta_4d.affine
    vox_size = np.abs(np.diag(affine)[:3])
    mask     = np.all(data != 0, axis=-1)
    coords   = np.array(np.where(mask)).T
    dists    = np.sqrt(((coords - center_vox)**2 * vox_size**2).sum(1))
    nb       = coords[dists <= radius_mm]
    return data[nb[:, 0], nb[:, 1], nb[:, 2], :].T   # (n_cond, k)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_variants(img):
    """
    Return (img_full, img_pos, img_top10) variants of a stat map.

    full  : original  (positive + negative, symmetric colorbar)
    pos   : positive values only (negatives → NaN)
    top10 : top 10% of positive values only (bottom 90% of positives → NaN)
    """
    data = img.get_fdata().copy()

    # pos
    pos_data = data.copy()
    pos_data[pos_data <= 0] = np.nan
    img_pos = nib.Nifti1Image(pos_data, img.affine, img.header)

    # top10
    pos_vals = data[data > 0]
    if len(pos_vals) > 0:
        thresh = np.nanpercentile(pos_vals, 90)
        top_data = data.copy()
        top_data[top_data <= thresh] = np.nan
    else:
        top_data = pos_data.copy()
    img_top = nib.Nifti1Image(top_data, img.affine, img.header)

    return img, img_pos, img_top


def _stat_map_row(img, ax, title, threshold=0.0, n_cuts=8):
    from nilearn import plotting
    data  = img.get_fdata()
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        ax.set_visible(False)
        return
    vmax = np.nanpercentile(np.abs(valid), 99)
    if vmax == 0 or np.isnan(vmax):
        ax.set_visible(False)
        return
    has_neg  = np.any(valid < 0)
    cmap     = "RdYlBu_r" if has_neg else "YlOrRd"
    vmin_arg = -vmax if has_neg else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotting.plot_stat_map(
            img, axes=ax,
            threshold=threshold, display_mode="z",
            cut_coords=n_cuts, colorbar=True,
            cmap=cmap, vmax=vmax, vmin=vmin_arg,
            title=title,
        )


# ── 7a. Whole-brain axial — 3 variants ───────────────────────────────────────

_VARIANTS = [
    ("full",  "full (pos + neg)"),
    ("pos",   "positive only"),
    ("top10", "top 10% positive"),
]


def plot_wholebrain(results_dir, plots_dir, subject,
                    prefix, method_label, model_names):
    """
    Saves 3 PNGs per prefix, one row per model in each:
      *_wholebrain_full.png
      *_wholebrain_pos.png
      *_wholebrain_top10.png
    """
    nii_paths = [
        os.path.join(results_dir, f"{subject}_{prefix}_{n}.nii.gz")
        for n in model_names
    ]
    nii_paths = [p for p in nii_paths if os.path.exists(p)]
    if not nii_paths:
        print(f"    [skip wholebrain] no maps for {subject}/{prefix}")
        return

    n = len(nii_paths)

    for var_tag, var_label in _VARIANTS:
        fig, axes = plt.subplots(n, 1, figsize=(20, 3.5 * n))
        if n == 1:
            axes = [axes]

        for ax, path in zip(axes, nii_paths):
            mname = (os.path.basename(path)
                     .replace(f"{subject}_{prefix}_", "")
                     .replace(".nii.gz", ""))
            img_full, img_pos, img_top = _prepare_variants(nib.load(path))
            img_plot = {"full": img_full, "pos": img_pos, "top10": img_top}[var_tag]
            _stat_map_row(img_plot, ax, f"{subject} | {mname} | {var_label}")

        fig.suptitle(f"{method_label} | {subject} | {var_label}",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = os.path.join(plots_dir,
                           f"{subject}_{prefix}_wholebrain_{var_tag}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out}")


# ── 7b. Region neural RDM + model scatter ────────────────────────────────────

def plot_region_rdm(results_dir, plots_dir, subject,
                    beta_test, test_conds, model_rdms_test,
                    region_name, region_mni, radius_mm):
    """
    Neural RDM figure for a spherical ROI.

    Layout
    ------
    Row 0      : raw neural RDM  |  ranked neural RDM
    Row 1..N   : model RDM       |  rank-scatter vs neural
    """
    center_vox = mni_to_vox(region_mni, beta_test.affine)
    patterns   = extract_sphere(beta_test, center_vox, radius_mm)
    n_cond     = len(test_conds)

    if patterns.shape[1] < n_cond:
        print(f"    [skip {region_name} RDM] too few voxels for {subject}")
        return

    neural_vec = pdist(patterns, metric="correlation")
    neural_rdm = squareform(neural_vec)
    ranked_rdm = squareform(rankdata(neural_vec))
    np.fill_diagonal(ranked_rdm, 0)

    short  = [short_label(c) for c in test_conds]
    n_mod  = len(model_rdms_test)
    n_rows = 1 + n_mod
    fig    = plt.figure(figsize=(14, 5 + 5 * n_mod))
    gs     = gridspec.GridSpec(n_rows, 2, hspace=0.5, wspace=0.35)

    # Row 0: raw + ranked neural RDM
    for col, (mat, title, cb_label, vmax) in enumerate([
        (neural_rdm, f"{region_name} Neural RDM (raw 1-r)",  "1-r",  2.0),
        (ranked_rdm, f"{region_name} Neural RDM (ranked)",   "rank",
         float(np.nanmax(ranked_rdm))),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=vmax,
                       interpolation="none", aspect="auto")
        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(short, fontsize=5, rotation=90)
        # ax.set_yticks(range(n_cond))
        # ax.set_yticklabels(short, fontsize=5)
        ax.set_title(title, fontweight="bold", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, label=cb_label)

    # Rows 1+: model RDM + scatter
    for row, (name, rdm) in enumerate(model_rdms_test.items()):
        mvec      = rdm_to_vec(rdm)
        rho, pval = spearmanr(neural_vec, mvec)
        rdm_full  = squareform(mvec) if mvec.ndim == 1 else rdm
        n_m       = rdm_full.shape[0]
        short_m   = [short_label(c) for c in test_conds[:n_m]]

        ax_m = fig.add_subplot(gs[1 + row, 0])
        im_m = ax_m.imshow(rdm_full, cmap="RdYlBu_r", vmin=0, vmax=1,
                            interpolation="none", aspect="auto")
        ax_m.set_xticks(range(n_m))
        ax_m.set_xticklabels(short_m, fontsize=5, rotation=90)
        # ax_m.set_yticks(range(n_m))
        # ax_m.set_yticklabels(short_m, fontsize=5)
        ax_m.set_title(f"Model: {name}  rho={rho:.3f}  p={pval:.3f}",
                       fontweight="bold", fontsize=9)
        plt.colorbar(im_m, ax=ax_m, fraction=0.046)

        ax_s = fig.add_subplot(gs[1 + row, 1])
        ax_s.scatter(rankdata(mvec), rankdata(neural_vec),
                     alpha=0.3, s=6, color="steelblue", rasterized=True)
        ax_s.set_xlabel(f"Model rank ({name})", fontsize=8)
        ax_s.set_ylabel("Neural rank", fontsize=8)
        ax_s.set_title(f"Rank scatter  rho={rho:.3f}", fontsize=9,
                       fontweight="bold")

    fig.suptitle(
        f"{subject} | {region_name} {region_mni}  r={radius_mm}mm",
        fontsize=12, fontweight="bold",
    )
    tag = region_name.lower()
    out = os.path.join(plots_dir, f"{subject}_{tag}_rdm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")


# ── 7c. DLPFC bar chart ───────────────────────────────────────────────────────

def plot_dlpfc_barchart(results_dir, plots_dir, subject,
                         prefixes, dlpfc_mni):
    rows = []
    for prefix in prefixes:
        for f in sorted(os.listdir(results_dir)):
            if not (f.startswith(f"{subject}_{prefix}_") and
                    f.endswith(".nii.gz")):
                continue
            model_name = (f.replace(f"{subject}_{prefix}_", "")
                           .replace(".nii.gz", ""))
            img  = nib.load(os.path.join(results_dir, f))
            cv   = mni_to_vox(dlpfc_mni, img.affine)
            data = img.get_fdata()
            val  = (float(data[cv[0], cv[1], cv[2]])
                    if all(0 <= cv[i] < data.shape[i] for i in range(3))
                    else np.nan)
            rows.append({"method": prefix, "model": model_name, "value": val})

    if not rows:
        print(f"    [skip barchart] no maps for {subject}")
        return

    df               = pd.DataFrame(rows)
    prefixes_present = df["method"].unique()
    models_present   = sorted(df["model"].unique())
    n_m  = len(prefixes_present)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4), sharey=False)
    if n_m == 1:
        axes = [axes]

    for ax, pref in zip(axes, prefixes_present):
        sub  = df[df["method"] == pref].set_index("model")
        vals = [float(sub.loc[m, "value"]) if m in sub.index else np.nan
                for m in models_present]
        colors = ["#d73027" if (v > 0 and not np.isnan(v)) else "#4575b4"
                  for v in vals]
        ax.bar(range(len(models_present)), vals,
               color=colors, edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(len(models_present)))
        ax.set_xticklabels(models_present, rotation=45, ha="right", fontsize=8)
        ax.set_title(pref, fontsize=9, fontweight="bold")
        ax.set_ylabel("Value at DLPFC")

    fig.suptitle(f"{subject} | DLPFC {dlpfc_mni}", fontsize=11,
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{subject}_dlpfc_barchart.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rsa_analysis.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            RSA searchlight analysis + visualization.
            Runs downstream of the LSS GLM pipeline.
        """),
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--subjects",    nargs="+", required=True)
    parser.add_argument("--radius-mm",   type=float, default=12.0)
    parser.add_argument("--use-ranks",   action="store_true")
    parser.add_argument("--n-jobs",      type=int, default=-1)
    # parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--dlpfc-coord", nargs=3, type=float,
                        default=[-44, 36, 28], metavar=("X","Y","Z"))
    parser.add_argument("--loc-coord",   nargs=3, type=float,
                        default=[-42, -74, -8], metavar=("X","Y","Z"),
                        help="MNI coord for LOC probe. (default: -42 -74 -8)")
    args = parser.parse_args(argv)

    plots_dir      = os.path.join(args.results_dir, "plots_ranked" if args.use_ranks else "plots")
    os.makedirs(plots_dir, exist_ok=True)
    dlpfc_mni      = tuple(args.dlpfc_coord)
    loc_mni        = tuple(args.loc_coord)
    ols_prefix_tag = "ranked" if args.use_ranks else "raw"

    print("=" * 60)
    print(f"  RSA analysis")
    print(f"  results_dir : {args.results_dir}")
    print(f"  subjects    : {args.subjects}")
    print(f"  radius_mm   : {args.radius_mm}")
    print(f"  use_ranks   : {args.use_ranks}")
    print(f"  dlpfc_mni   : {dlpfc_mni}")
    print(f"  loc_mni     : {loc_mni}")
    print("=" * 60)

    for subject in args.subjects:
        print(f"\n{'─'*60}\n  {subject}\n{'─'*60}")

        # Load LSS betas
        print("  loading LSS betas ...")
        beta_4d, conditions = load_lss_betas(args.results_dir, subject)
        beta_test, test_conds = subset_to_test(beta_4d, conditions)
        print(f"  full conditions : {len(conditions)}")
        print(f"  test conditions : {len(test_conds)}")

        # Build + save model RDMs
        print("  building model RDMs ...")
        rdms_full = make_model_rdms_full(conditions)
        rdms_test = make_model_rdms_test(conditions)
        save_model_rdms(args.results_dir, "full", conditions, rdms_full)
        save_model_rdms(args.results_dir, "test", test_conds, rdms_test)
        print(f"  full models : {list(rdms_full.keys())}")
        print(f"  test models : {list(rdms_test.keys())}")

        # Searchlights
        runs = [
            ("full", beta_4d,   rdms_full),
            ("test", beta_test, rdms_test),
        ]
        for tag, b4d, rdms in runs:

            sp_prefix = f"spearman_{tag}"
            # if not args.overwrite and maps_cached(
            #         args.results_dir, subject, sp_prefix, list(rdms.keys())):
            #     print(f"  [cache] {sp_prefix}")
            print(f"  running spearman/{tag} ...")
            arr, names, vcoords, aff, vshape = run_spearman_searchlight(
                b4d, rdms, radius_mm=args.radius_mm, n_jobs=args.n_jobs)
            save_maps(args.results_dir, subject, sp_prefix,
                        arr, names, vcoords, aff, vshape)

            ols_prefix = f"ols_{ols_prefix_tag}_{tag}"
            # if not args.overwrite and maps_cached(
            #         args.results_dir, subject, ols_prefix, list(rdms.keys())):
            #     print(f"  [cache] {ols_prefix}")
            # else:
            print(f"  running ols/{ols_prefix_tag}/{tag} ...")
            arr, names, vcoords, aff, vshape = run_ols_searchlight(
                b4d, rdms, radius_mm=args.radius_mm,
                use_ranks=args.use_ranks, n_jobs=args.n_jobs)
            save_maps(args.results_dir, subject, ols_prefix,
                        arr, names, vcoords, aff, vshape)

        # Plots
        all_prefixes = [
            "spearman_full", "spearman_test",
            f"ols_{ols_prefix_tag}_full", f"ols_{ols_prefix_tag}_test",
        ]

        print("  plotting whole-brain maps (3 variants each) ...")
        for prefix in all_prefixes:
            model_names_for_prefix = (list(rdms_full.keys())
                                      if "full" in prefix
                                      else list(rdms_test.keys()))
            plot_wholebrain(args.results_dir, plots_dir, subject,
                            prefix, prefix, model_names_for_prefix)

        print("  plotting region RDMs ...")
        for region_name, region_mni in [("DLPFC", dlpfc_mni),
                                         ("LOC",   loc_mni)]:
            plot_region_rdm(args.results_dir, plots_dir, subject,
                            beta_test, test_conds, rdms_test,
                            region_name, region_mni, args.radius_mm)

        print("  plotting DLPFC bar chart ...")
        plot_dlpfc_barchart(args.results_dir, plots_dir, subject,
                            all_prefixes, dlpfc_mni)

        print(f"  {subject} complete.")

    print(f"\nAll done. Plots -> {plots_dir}")


if __name__ == "__main__":
    main()
