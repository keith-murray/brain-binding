"""RSA pipeline with LSS-based GLM.

Pipeline
--------
1. LSS GLM  → trial-level betas → average within condition → beta_4d
2. Model RDMs (full set + test-phase variants)
3. Searchlight: simple Spearman rho  (per model)
4. Searchlight: raw OLS in RDM space (partial regression, no ranking)
5. Save NIfTI maps + PNG brain heatmaps

Caching
-------
Every expensive output (LSS betas, searchlight maps) is written to disk.
On re-run the script checks for existing files and skips recomputation
unless --overwrite is passed.

Acceleration
------------
* LSS inner loop: Parallel over trials via joblib (n_jobs configurable)
* Searchlight:    Parallel over voxels
* GLM:            minimize_memory=True, signal_scaling=False
* Beta averaging: in-memory accumulation (no repeated disk I/O)

Usage
-----
python rsa_lss.py --subject-table subjects.csv --results-dir rsa_results

Subject table columns (CSV/TSV):
  subject, bold_path, behavioral_csv, log_path
  [optional] mask_path | probseg_path
"""

from __future__ import annotations

import warnings
import argparse
import os
import textwrap
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1.  EVENT BUILDING
# ══════════════════════════════════════════════════════════════════════════════
def load_confounds_manual(bold_path, strategy=None, fd_threshold=0.5,
                           n_compcor=5, motion="basic", demean=False):
    """
    Drop-in replacement for nilearn's load_confounds.
    Infers the TSV path from the BOLD path (same logic as load_confounds)
    but bypasses the version check that rejects older fMRIPrep outputs.
    """
    import re

    # Infer TSV path: strip space/res entities, swap desc-preproc_bold -> desc-confounds_timeseries
    tsv_path = re.sub(
        r"_space-[^_]+",  "", bold_path   # remove _space-XXX
    )
    tsv_path = re.sub(
        r"_res-[^_]+",    "", tsv_path    # remove _res-X
    )
    tsv_path = tsv_path.replace(
        "_desc-preproc_bold.nii.gz",
        "_desc-confounds_timeseries.tsv"
    )

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"Confounds TSV not found: {tsv_path}\n"
            f"  (inferred from: {bold_path})"
        )

    print(f"  confounds TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Motion parameters
    if motion == "basic":
        motion_cols = [c for c in ["trans_x","trans_y","trans_z",
                                    "rot_x","rot_y","rot_z"] if c in df.columns]
    else:  # "full" — include derivatives and squares
        motion_cols = [c for c in df.columns
                       if any(c.startswith(p)
                              for p in ["trans_x","trans_y","trans_z",
                                        "rot_x","rot_y","rot_z"])]

    # aCompCor
    acompcor_cols = sorted([c for c in df.columns
                             if c.startswith("a_comp_cor_")])[:n_compcor]

    # WM + CSF
    wm_csf_cols = [c for c in ["white_matter","csf"] if c in df.columns]

    # Global signal
    gs_cols = [c for c in ["global_signal"] if c in df.columns]

    # High-pass cosines
    cosine_cols = [c for c in df.columns if c.startswith("cosine")]

    # Spike regressors from fMRIPrep
    outlier_cols = sorted([c for c in df.columns
                            if c.startswith("motion_outlier")])

    # Assemble based on strategy
    strat = strategy or ["motion","wm_csf","global_signal","compcor",
                          "scrub","high_pass"]
    all_cols = []
    if "motion"        in strat: all_cols += motion_cols
    if "compcor"       in strat: all_cols += acompcor_cols
    if "wm_csf"        in strat: all_cols += wm_csf_cols
    if "global_signal" in strat: all_cols += gs_cols
    if "high_pass"     in strat: all_cols += cosine_cols
    if "scrub"         in strat: all_cols += outlier_cols

    all_cols = [c for c in all_cols if c in df.columns]
    confounds_df = df[all_cols].fillna(0)

    if demean:
        confounds_df = confounds_df - confounds_df.mean()

    # Sample mask from FD
    if "framewise_displacement" in df.columns:
        fd = df["framewise_displacement"].fillna(0)
        good = np.where(fd < fd_threshold)[0]
        sample_mask = good if len(good) < len(fd) else None
    else:
        sample_mask = None

    print(f"  confounds: {len(motion_cols)} motion, {len(acompcor_cols)} compcor, "
          f"{len(outlier_cols)} spikes, {len(cosine_cols)} cosines")
    print(f"  frames kept: "
          f"{len(sample_mask) if sample_mask is not None else len(df)}/{len(df)}")

    return confounds_df, sample_mask

def get_scan_start_time(log_path: str) -> float:
    with open(log_path) as f:
        for line in f:
            if "Keypress: equal" in line:
                return float(line.split("\t")[0])
    raise ValueError(f"No scanner TTL in {log_path}")


def build_stimulus_events(
    behavioral_csv_path: str,
    log_path: str,
    stim_duration: float = 1.0,
    main_block_prefix: str = "main",
) -> pd.DataFrame:
    """Trial-level CSV → stimulus-level events DataFrame for nilearn GLM."""
    df = pd.read_csv(behavioral_csv_path)
    df_main = df[df["block"].astype(str).str.startswith(main_block_prefix)].copy()
    scan_t0 = get_scan_start_time(log_path)

    rows: list[dict] = []
    for _, trial in df_main.iterrows():
        rule_type = str(trial["trial_type"])
        A, B, Ap, Bp = trial["A_stim"], trial["B_stim"], trial["A_prime"], trial["B_prime"]

        for t, shape, role, phase in [
            (trial["t_rule1"], A,  "A",  "rule1"),
            (trial["t_rule2"], B,  "B",  "rule2"),
            (trial["t_rule3"],
             A if rule_type == "ABA" else B,
             "A" if rule_type == "ABA" else "B",
             "rule3"),
        ]:
            rows.append({"onset": float(t) - scan_t0,
                         "duration": stim_duration,
                         "trial_type": f"{shape}_{role}_{phase}"})

        for t, shape, role, phase in [
            (trial["t_test1"], Ap, "Aprime", "test1"),
            (trial["t_test2"], Bp, "Bprime", "test2"),
        ]:
            rows.append({"onset": float(t) - scan_t0,
                         "duration": stim_duration,
                         "trial_type": f"{shape}_{role}_{phase}_{rule_type}"})

    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def parse_condition(label: str) -> dict:
    parts = str(label).split("_")
    shape, role, phase = parts[0], parts[1], parts[2]
    rule_type = parts[3] if len(parts) == 4 else None
    return {"shape": shape, "role": role, "phase": phase,
            "rule_type": rule_type,
            "epoch": "rule" if phase.startswith("rule") else "test"}


def build_condition_metadata(conditions: Iterable[str]) -> pd.DataFrame:
    conds = list(conditions)
    return pd.DataFrame([parse_condition(c) for c in conds], index=conds)


def filter_test_conditions(conditions: list[str]) -> list[str]:
    return [c for c in conditions if parse_condition(c)["phase"].startswith("test")]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LSS GLM
# ══════════════════════════════════════════════════════════════════════════════

def _fit_one_lss_trial(
    trial_idx: int,
    events_df: pd.DataFrame,
    bold_path: str,
    mask_img,
    sample_mask: np.ndarray,
    t_r: float,
    n_trs: int,
    confounds,
) -> tuple[str, np.ndarray]:
    """Fit one LSS GLM for trial `trial_idx`. Returns (trial_type, flat_beta_array)."""
    import nilearn as nl
    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
    import nibabel as nib

    trial = events_df.iloc[trial_idx]
    trial_type = str(trial["trial_type"])

    # This trial gets its own regressor; everything else → 'other'
    this_event = events_df.iloc[[trial_idx]].copy()
    other_events = events_df.drop(index=trial_idx).copy()
    other_events["trial_type"] = "other"
    lss_events = pd.concat([this_event, other_events], ignore_index=True)

    TR_array = np.arange(n_trs) * t_r
    dm = make_first_level_design_matrix(
        TR_array, lss_events, hrf_model="glover",
        drift_model=None,
        add_regs=confounds,
    )

    bold_img = nib.load(bold_path)
    model = FirstLevelModel(
        t_r=t_r, hrf_model="glover", mask_img=mask_img,
        smoothing_fwhm=None, standardize="zscore_sample",
        signal_scaling=False, drift_model=None, minimize_memory=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit([bold_img], design_matrices=[dm], sample_masks=[sample_mask])
        beta_img = model.compute_contrast(trial_type, output_type="effect_size")
    return trial_type, beta_img.get_fdata().ravel()


def run_lss_and_average(
    bold_path: str,
    events_df: pd.DataFrame,
    mask_img,
    t_r: float = 2.0,
    confound_strategy: list[str] | None = None,
    fd_threshold: float = 0.5,
    n_compcor: int = 5,
    motion: str = "basic",
    demean_confounds: bool = False,
    n_jobs: int = -1,
    verbose: int = 5,
) -> tuple:
    """
    Run LSS (one GLM per trial) and return averaged beta_4d + condition list.

    Returns
    -------
    beta_4d   : nibabel.Nifti1Image  shape (x,y,z, n_conditions)
    conditions: list[str]
    """
    import nibabel as nib
    import nilearn as nl
    from nilearn.image import concat_imgs
    from joblib import Parallel, delayed

    bold_img = nib.load(bold_path)
    n_trs = bold_img.shape[-1]
    brain_shape = bold_img.shape[:3]
    affine = bold_img.affine

    if confound_strategy is None:
        confound_strategy = ["motion", "wm_csf", "global_signal",
                             "compcor", "scrub", "high_pass"]

    # from nilearn.interfaces.fmriprep import load_confounds

    # confounds, sample_mask = load_confounds(
    #     bold_path,
    #     strategy=confound_strategy,
    #     fd_threshold=fd_threshold,
    #     n_compcor=n_compcor,
    #     motion=motion,
    #     demean=demean_confounds,
    # )
    confounds, sample_mask = load_confounds_manual(
        bold_path,
        strategy     = confound_strategy,
        fd_threshold = fd_threshold,
        n_compcor    = n_compcor,
        motion       = motion,
        demean       = demean_confounds,
    )
    if sample_mask is None:
        sample_mask = np.arange(n_trs)

    # Resampled mask — needed inside worker
    if mask_img is not None:
        mask_img_r = nl.image.resample_to_img(mask_img, bold_img, interpolation="nearest")
    else:
        mask_img_r = None

    n_trials = len(events_df)
    print(f"  LSS: {n_trials} trials, parallel n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_one_lss_trial)(
            i, events_df, bold_path, mask_img_r, sample_mask,
            t_r, n_trs, confounds,
        )
        for i in range(n_trials)
    )

    # Accumulate: sum betas per condition + count
    conditions_set = sorted({r[0] for r in results})
    n_cond = len(conditions_set)
    n_vox  = int(np.prod(brain_shape))
    cond_index = {c: i for i, c in enumerate(conditions_set)}

    beta_sum   = np.zeros((n_cond, n_vox), dtype=np.float32)
    beta_count = np.zeros(n_cond, dtype=np.int32)

    for trial_type, flat_beta in results:
        idx = cond_index[trial_type]
        beta_sum[idx]   += flat_beta
        beta_count[idx] += 1

    beta_avg = beta_sum / np.maximum(beta_count[:, None], 1)  # (n_cond, n_vox)

    # Stack into 4D NIfTI: (x, y, z, n_cond)
    vol_list = []
    for i in range(n_cond):
        vol = beta_avg[i].reshape(brain_shape)
        vol_list.append(nib.Nifti1Image(vol.astype(np.float32), affine))

    beta_4d = concat_imgs(vol_list)
    return beta_4d, conditions_set


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_beta_4d(results_dir: str, subject: str, beta_4d, conditions: list[str]):
    os.makedirs(results_dir, exist_ok=True)
    beta_path   = os.path.join(results_dir, f"{subject}_lss_beta_4d.nii.gz")
    labels_path = os.path.join(results_dir, f"{subject}_lss_conditions.csv")
    beta_4d.to_filename(beta_path)
    pd.Series(conditions).to_csv(labels_path, index=False)
    return beta_path, labels_path


def load_beta_4d(results_dir: str, subject: str):
    import nibabel as nib
    beta_path   = os.path.join(results_dir, f"{subject}_lss_beta_4d.nii.gz")
    labels_path = os.path.join(results_dir, f"{subject}_lss_conditions.csv")
    beta_4d     = nib.load(beta_path)
    conditions  = pd.read_csv(labels_path).iloc[:, 0].tolist()
    return beta_4d, conditions


def beta_cache_exists(results_dir: str, subject: str) -> bool:
    return (
        os.path.exists(os.path.join(results_dir, f"{subject}_lss_beta_4d.nii.gz")) and
        os.path.exists(os.path.join(results_dir, f"{subject}_lss_conditions.csv"))
    )


def searchlight_cache_exists(results_dir: str, subject: str, prefix: str, model_names: list[str]) -> bool:
    return all(
        os.path.exists(os.path.join(results_dir, f"{subject}_{prefix}_{n}.nii.gz"))
        for n in model_names
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL RDMs
# ══════════════════════════════════════════════════════════════════════════════

def abstract_role(role: str) -> str:
    return "A" if role in {"A", "Aprime"} else "B"


def _make_rdm(conditions, meta, sim_fn) -> np.ndarray:
    n = len(conditions)
    rdm = np.zeros((n, n))
    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            rdm[i, j] = 1.0 - float(sim_fn(meta.loc[ci], meta.loc[cj]))
    np.fill_diagonal(rdm, 0.0)
    return rdm


def make_model_rdms_full(conditions: list[str]) -> dict[str, np.ndarray]:
    meta = build_condition_metadata(conditions)

    return {
        "M1_shape":         _make_rdm(conditions, meta,
                                lambda a,b: float(a["shape"]==b["shape"])),
        "M2_abstract_role": _make_rdm(conditions, meta,
                                lambda a,b: float(abstract_role(a["role"])==abstract_role(b["role"]))),
        "M3_concrete_role": _make_rdm(conditions, meta,
                                lambda a,b: float(a["role"]==b["role"])),
        "M4_epoch":         _make_rdm(conditions, meta,
                                lambda a,b: float(a["epoch"]==b["epoch"])),
        "M5_rule_type":     _make_rdm(conditions, meta,
                                lambda a,b: (0.5 if a["rule_type"] is None or b["rule_type"] is None
                                             else float(a["rule_type"]==b["rule_type"]))),
        "M6_binding":       _make_rdm(conditions, meta,
                                lambda a,b: float(a["shape"]==b["shape"] and
                                                  abstract_role(a["role"])==abstract_role(b["role"]))),
    }


def make_model_rdms_test(conditions: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    test_conds = filter_test_conditions(conditions)
    meta = build_condition_metadata(test_conds)

    def is_repeater(row):
        return row["role"] == "Aprime" if row["rule_type"] == "ABA" else row["role"] == "Bprime"

    rdms = {
        "shape_test":      _make_rdm(test_conds, meta, lambda a,b: float(a["shape"]==b["shape"])),
        "abstract_test":   _make_rdm(test_conds, meta, lambda a,b: float(a["role"]==b["role"])),
        "functional_test": _make_rdm(test_conds, meta, lambda a,b: float(is_repeater(a)==is_repeater(b))),
    }
    return test_conds, rdms


def rdm_to_vec(rdm: np.ndarray) -> np.ndarray:
    """Full (n,n) matrix → condensed (n*(n-1)/2,) vector matching pdist order."""
    from scipy.spatial.distance import squareform
    return squareform(rdm) if rdm.ndim == 2 else rdm


def save_model_rdms(results_dir: str, tag: str, conditions: list[str],
                    model_rdms: dict[str, np.ndarray]):
    path = os.path.join(results_dir, f"model_rdms_{tag}.npz")
    np.savez(path, conditions=np.array(conditions),
             **{k: v for k, v in model_rdms.items()})
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SEARCHLIGHT CORE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SearchlightSetup:
    beta_data:   np.ndarray   # (x,y,z,n_cond)
    affine:      np.ndarray   # (4,4)
    vox_size_mm: np.ndarray   # (3,)
    vox_coords:  np.ndarray   # (n_vox,3)
    vol_shape:   tuple


def prepare_searchlight(beta_4d) -> SearchlightSetup:
    data  = beta_4d.get_fdata()
    aff   = beta_4d.affine
    vsz   = np.abs(np.diag(aff)[:3])
    mask  = np.all(data != 0, axis=-1)
    coords = np.array(np.where(mask)).T
    return SearchlightSetup(beta_data=data, affine=aff,
                            vox_size_mm=vsz, vox_coords=coords,
                            vol_shape=data.shape[:3])


def _sphere_patterns(center_idx, beta_data, vox_coords, vox_size_mm, radius_mm, n_cond):
    """Return (neural_vec, ok) for a single searchlight center."""
    from scipy.spatial.distance import pdist
    cv  = vox_coords[center_idx]
    d   = np.sqrt(((vox_coords - cv)**2 * vox_size_mm**2).sum(1))
    nb  = np.where(d <= radius_mm)[0]
    if len(nb) < n_cond:
        return None, False
    nv       = vox_coords[nb]
    patterns = beta_data[nv[:,0], nv[:,1], nv[:,2], :].T   # (n_cond, k)
    return pdist(patterns, metric="correlation"), True       # (vec_len,)


# ── 5a.  Simple Spearman RSA ──────────────────────────────────────────────────

def _spearman_voxel(center_idx, beta_data, vox_coords, vox_size_mm,
                    model_vecs, model_names, n_cond, radius_mm):
    from scipy.stats import spearmanr
    try:
        neural_vec, ok = _sphere_patterns(center_idx, beta_data, vox_coords,
                                          vox_size_mm, radius_mm, n_cond)
        if not ok:
            return np.full(len(model_names), np.nan)
        return np.array([spearmanr(neural_vec, model_vecs[n])[0]
                         for n in model_names], dtype=float)
    except Exception:
        return np.full(len(model_names), np.nan)


def run_spearman_searchlight(beta_4d, model_rdms: dict, radius_mm=12.0,
                              n_jobs=-1, verbose=5):
    from joblib import Parallel, delayed
    sl     = prepare_searchlight(beta_4d)
    n_cond = sl.beta_data.shape[-1]
    names  = list(model_rdms.keys())
    vecs   = {n: rdm_to_vec(rdm) for n, rdm in model_rdms.items()}

    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_spearman_voxel)(
            i, sl.beta_data, sl.vox_coords, sl.vox_size_mm,
            vecs, names, n_cond, radius_mm)
        for i in range(len(sl.vox_coords))
    )
    arr = _sanitize(res, len(names))
    return arr, names, sl.vox_coords, sl.affine, sl.vol_shape


# ── 5b.  Raw OLS in RDM space (no ranking) ───────────────────────────────────
# y  = neural distance vector  (raw correlation distances, continuous)
# X  = model RDM vectors       (binary {0,1}, mean-centred)
# β_k = mean neural-distance difference for model k, partialling out others

def _ols_voxel(center_idx, beta_data, vox_coords, vox_size_mm,
               model_vecs, model_names, n_cond, radius_mm):
    from numpy.linalg import lstsq
    try:
        neural_vec, ok = _sphere_patterns(center_idx, beta_data, vox_coords,
                                          vox_size_mm, radius_mm, n_cond)
        if not ok:
            return np.full(len(model_names), np.nan)

        # y: raw correlation distances (no ranking)
        y = neural_vec.reshape(-1, 1)

        # X: binary model vectors, mean-centred so intercept = grand mean
        X = np.column_stack([model_vecs[n].astype(float) - model_vecs[n].mean()
                              for n in model_names])
        X = np.hstack([X, np.ones((X.shape[0], 1))])   # intercept last

        betas, _, _, _ = lstsq(X, y, rcond=None)
        return np.asarray(betas[:-1, 0], dtype=float)   # drop intercept
    except Exception:
        return np.full(len(model_names), np.nan)


def run_ols_searchlight(beta_4d, model_rdms: dict, radius_mm=12.0,
                         n_jobs=-1, verbose=5):
    from joblib import Parallel, delayed
    sl     = prepare_searchlight(beta_4d)
    n_cond = sl.beta_data.shape[-1]
    names  = list(model_rdms.keys())
    vecs   = {n: rdm_to_vec(rdm) for n, rdm in model_rdms.items()}

    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_ols_voxel)(
            i, sl.beta_data, sl.vox_coords, sl.vox_size_mm,
            vecs, names, n_cond, radius_mm)
        for i in range(len(sl.vox_coords))
    )
    arr = _sanitize(res, len(names))
    return arr, names, sl.vox_coords, sl.affine, sl.vol_shape


def _sanitize(res_list, n_models):
    out = []
    for r in res_list:
        r = np.asarray(r, dtype=float)
        out.append(r if r.shape == (n_models,) else np.full(n_models, np.nan))
    return np.array(out)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE NIfTI MAPS + PNG HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

def save_maps(results_dir, subject, prefix, results_arr, model_names,
              vox_coords, affine, vol_shape):
    import nibabel as nib
    os.makedirs(results_dir, exist_ok=True)
    paths = {}
    for m_idx, name in enumerate(model_names):
        vol = np.full(vol_shape, np.nan, dtype=float)
        vol[vox_coords[:,0], vox_coords[:,1], vox_coords[:,2]] = results_arr[:, m_idx]
        img = nib.Nifti1Image(vol, affine)
        p   = os.path.join(results_dir, f"{subject}_{prefix}_{name}.nii.gz")
        img.to_filename(p)
        paths[name] = p
        print(f"  saved {p}  mean={np.nanmean(vol):.4f}  max={np.nanmax(vol):.4f}")
    return paths


def save_heatmaps(results_dir, subject, prefix, model_names,
                  cut_coords=(0, -52, 28), threshold=0.0):
    """Save one ortho PNG per model map."""
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting, image

    for name in model_names:
        nii_path = os.path.join(results_dir, f"{subject}_{prefix}_{name}.nii.gz")
        if not os.path.exists(nii_path):
            print(f"  [heatmap] missing {nii_path}, skipping")
            continue

        img    = nib.load(nii_path)
        smooth = image.smooth_img(img, fwhm=6)          # display only
        data   = smooth.get_fdata()
        vmax   = np.nanmax(np.abs(data))
        if vmax == 0 or np.isnan(vmax):
            continue

        fig = plt.figure(figsize=(10, 3))
        display = plotting.plot_stat_map(
            smooth,
            threshold    = threshold,
            display_mode = "ortho",
            cut_coords   = cut_coords,
            colorbar     = True,
            cmap         = "RdYlBu_r",
            vmax         = vmax,
            figure       = fig,
            title        = f"{subject} | {prefix} | {name}",
        )
        out_png = os.path.join(results_dir, f"{subject}_{prefix}_{name}.png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved heatmap {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MASK UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def binarize_probseg(probseg_path: str, threshold: float = 0.4):
    import nibabel as nib
    img  = nib.load(probseg_path)
    data = img.get_fdata()
    return nib.Nifti1Image((data > threshold).astype(np.uint8), img.affine, img.header)


def _resolve_mask(row: pd.Series, threshold: float):
    mask_path    = row.get("mask_path")
    probseg_path = row.get("probseg_path")
    has_mask     = isinstance(mask_path, str) and str(mask_path).strip()
    has_probseg  = isinstance(probseg_path, str) and str(probseg_path).strip()
    if has_mask and has_probseg:
        raise ValueError(f"Subject {row['subject']}: provide only one of mask_path/probseg_path")
    if has_mask:
        import nibabel as nib
        return nib.load(mask_path)
    if has_probseg:
        return binarize_probseg(probseg_path, threshold=threshold)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rsa_lss.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            LSS-based RSA searchlight pipeline.

            Runs for each subject:
              1.  LSS GLM  → condition-averaged beta_4d  (cached)
              2.  Model RDMs: full (M1-M6) + test-phase (shape/abstract/functional)
              3.  Spearman RSA searchlight  (per model, simple rho)
              4.  Raw OLS searchlight in RDM space  (partial regression, no ranking)
              5.  NIfTI maps + PNG heatmaps for every model × method

            Subject table (CSV/TSV) required columns:
              subject, bold_path, behavioral_csv, log_path
            Optional columns:
              mask_path | probseg_path
        """),
    )
    parser.add_argument("--subject-table", required=True)
    parser.add_argument("--results-dir",   default="rsa_results")
    parser.add_argument("--stim-duration", type=float, default=1.0)
    parser.add_argument("--tr",            type=float, default=2.0)
    parser.add_argument("--radius-mm",     type=float, default=12.0)
    parser.add_argument("--probseg-threshold", type=float, default=0.3)
    parser.add_argument("--n-jobs",        type=int,   default=-1,
                        help="Parallel workers (-1 = all cores). "
                             "Applied to both LSS and searchlight loops.")
    parser.add_argument("--overwrite",     action="store_true",
                        help="Recompute even if cached outputs exist.")
    parser.add_argument("--heatmap-coord", nargs=3, type=float,
                        default=[0, -52, 28],
                        metavar=("X","Y","Z"),
                        help="MNI coordinate for ortho heatmap cross-hair.")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    sep   = "\t" if args.subject_table.endswith(".tsv") else ","
    subjs = pd.read_csv(args.subject_table, sep=sep)
    cut   = tuple(args.heatmap_coord)

    for _, row in subjs.iterrows():
        subject  = str(row["subject"])
        bold_path = str(row["bold_path"])
        print(f"\n{'='*60}\n  {subject}\n{'='*60}")

        # ── Step 1: LSS betas (cached) ────────────────────────────────────
        if not args.overwrite and beta_cache_exists(args.results_dir, subject):
            print("  [cache] loading existing LSS betas ...")
            beta_4d, conditions = load_beta_4d(args.results_dir, subject)
        else:
            print("  [LSS] building stimulus events ...")
            events_df = build_stimulus_events(
                str(row["behavioral_csv"]), str(row["log_path"]),
                stim_duration=args.stim_duration,
            )
            mask_img  = _resolve_mask(row, args.probseg_threshold)
            print(f"  [LSS] fitting {len(events_df)} trial GLMs ...")
            beta_4d, conditions = run_lss_and_average(
                bold_path=bold_path,
                events_df=events_df,
                mask_img=mask_img,
                t_r=args.tr,
                n_jobs=args.n_jobs,
            )
            save_beta_4d(args.results_dir, subject, beta_4d, conditions)
            print(f"  [LSS] saved beta_4d {beta_4d.shape}, {len(conditions)} conditions")

        # ── Step 2: Model RDMs ────────────────────────────────────────────
        # Full set  (all 32 conditions)
        rdms_full = make_model_rdms_full(conditions)
        save_model_rdms(args.results_dir, "full", conditions, rdms_full)

        # Test-phase set  (16 conditions)
        from nilearn.image import index_img
        test_conds  = filter_test_conditions(conditions)
        test_indices = [conditions.index(c) for c in test_conds]
        beta_test    = index_img(beta_4d, test_indices)
        _, rdms_test = make_model_rdms_test(conditions)
        save_model_rdms(args.results_dir, "test", test_conds, rdms_test)

        print(f"  [models] full={list(rdms_full.keys())}")
        print(f"  [models] test={list(rdms_test.keys())}")

        # ── Step 3: Spearman searchlight ──────────────────────────────────
        for tag, b4d, rdms in [
            ("full",  beta_4d,  rdms_full),
            ("test",  beta_test, rdms_test),
        ]:
            prefix = f"spearman_{tag}"
            if not args.overwrite and searchlight_cache_exists(
                    args.results_dir, subject, prefix, list(rdms.keys())):
                print(f"  [cache] {prefix} maps exist, skipping")
            else:
                print(f"  [spearman/{tag}] running searchlight ...")
                arr, names, vcoords, aff, vshape = run_spearman_searchlight(
                    b4d, rdms, radius_mm=args.radius_mm, n_jobs=args.n_jobs)
                save_maps(args.results_dir, subject, prefix,
                          arr, names, vcoords, aff, vshape)
            save_heatmaps(args.results_dir, subject, prefix,
                          list(rdms.keys()), cut_coords=cut)

        # ── Step 4: Raw OLS searchlight ───────────────────────────────────
        for tag, b4d, rdms in [
            ("full",  beta_4d,   rdms_full),
            ("test",  beta_test, rdms_test),
        ]:
            prefix = f"ols_{tag}"
            if not args.overwrite and searchlight_cache_exists(
                    args.results_dir, subject, prefix, list(rdms.keys())):
                print(f"  [cache] {prefix} maps exist, skipping")
            else:
                print(f"  [ols/{tag}] running searchlight ...")
                arr, names, vcoords, aff, vshape = run_ols_searchlight(
                    b4d, rdms, radius_mm=args.radius_mm, n_jobs=args.n_jobs)
                save_maps(args.results_dir, subject, prefix,
                          arr, names, vcoords, aff, vshape)
            save_heatmaps(args.results_dir, subject, prefix,
                          list(rdms.keys()), cut_coords=cut)

        print(f"\n  {subject} complete.")

    print("\nAll subjects done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())