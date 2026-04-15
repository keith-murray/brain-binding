"""
plot_rsa_results.py
-------------------
Standalone visualization of RSA searchlight results.

Produces per subject:
  1. Whole-brain z-plane stat maps       (masked to brain, all searchlight NIfTIs)
  2. Neural RDM probe figures            (raw + ranked + model scatter)
     - DLPFC  x test-phase conditions
     - DLPFC  x ALL conditions  (full 32)
     - LOC    x test-phase conditions
     - LOC    x ALL conditions
  3. Searchlight value bar chart at DLPFC and LOC

LOC (Lateral Occipital Complex) is the canonical shape-selective visual area.
Bilateral peak from Kourtzi & Kanwisher 2000 / Grill-Spector et al. 2001:
  Left  LOC: (-42, -74, -8)
  Right LOC: ( 42, -74, -8)
We probe the left hemisphere by default (change LOC_MNI in CONFIG).

Usage
-----
python plot_rsa_results.py

Edit the CONFIG block to match your paths.
"""

import os
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

# ==============================================================================
# CONFIG -- edit these
# ==============================================================================

SCRIPT_DIR  = "/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "rsa_results_masked")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

SUBJECTS  = ["sub-001", "sub-002", "sub-008"]
RADIUS_MM = 12.0

# Gray matter masks -- per-subject GM probability segmentation from fMRIPrep.
# Thresholded at GM_THRESHOLD to create a binary GM mask, matching the notebook:
#   (probseg > threshold).astype(uint8)
# Files are in anat/ as: {sub}_ses-01_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz
GM_THRESHOLD = 0.3   # matches notebook default; range 0.2-0.5

FMRIPREP_DIR = ("/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding/"
                "data/bids/derivatives/fmriprep")

BRAIN_MASKS = {
    "sub-001": "/jukebox/graziano/kirsten/502b_percept_new/pygers_workshop/sample_study/data/bids/derivatives/fmriprep/sub-001/ses-01/anat/sub-001_ses-01_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz",
    "sub-002": "/jukebox/graziano/kirsten/502b_percept_new/pygers_workshop/sample_study/data/bids/derivatives/fmriprep/sub-002/ses-01/anat/sub-002_ses-01_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz",
    "sub-008": "/usr/people/gw0402/pygers_workshop/sample_study/data/bids/derivatives/fmriprep/sub-008/ses-01/anat/sub-008_ses-01_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz",
    }

# Probe coordinates
DLPFC_MNI = (-44,  36,  28)   # left DLPFC
LOC_MNI   = (-42, -74,  -8)   # left LOC -- shape-selective visual area

# Searchlight map prefixes present in RESULTS_DIR and their display labels
METHODS = {
    "spearman_full": "Spearman rho -- all conditions",
    "spearman_test": "Spearman rho -- test phase",
    "ols_raw_full":  "OLS beta (raw) -- all conditions",
    "ols_raw_test":  "OLS beta (raw) -- test phase",
}

# ==============================================================================
# HELPERS
# ==============================================================================

def parse_condition(label):
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


def short_label(cond):
    p    = parse_condition(cond)
    rule = f"_{p['rule_type']}" if p["rule_type"] else ""
    return f"{p['shape'][:3]}\n{p['role'][:3]}{rule}"


def mni_to_vox(mni_coord, affine):
    inv = np.linalg.inv(affine)
    vox = inv[:3, :3] @ np.array(mni_coord) + inv[:3, 3]
    return np.round(vox).astype(int)


def apply_gm_mask(stat_img, probseg_path, threshold=GM_THRESHOLD):
    """
    Apply a gray matter mask to a stat image.

    Loads the fMRIPrep GM probability segmentation, thresholds at
    `threshold` (same approach as the notebook), resamples to the
    stat image grid, and sets non-GM voxels to NaN.

    Returns a new NIfTI image safe for plotting.
    """
    if probseg_path is None or not os.path.exists(probseg_path):
        if probseg_path is not None:
            print(f"  WARNING: GM probseg not found: {probseg_path}")
        return stat_img

    from nilearn.image import resample_to_img

    # Threshold probability map -> binary GM mask  (matches notebook)
    prob_data = nib.load(probseg_path).get_fdata()
    prob_affine = nib.load(probseg_path).affine
    gm_binary = (prob_data > threshold).astype(np.uint8)
    gm_img    = nib.Nifti1Image(gm_binary, prob_affine)

    # Resample GM mask to stat image grid
    gm_resampled = resample_to_img(gm_img, stat_img, interpolation="nearest")
    gm_data      = gm_resampled.get_fdata().astype(bool)

    stat_data            = stat_img.get_fdata().copy()
    stat_data[~gm_data]  = np.nan
    return nib.Nifti1Image(stat_data, stat_img.affine, stat_img.header)


def load_lss_betas_full(results_dir, subject):
    beta_path   = os.path.join(results_dir, f"{subject}_lss_beta_4d.nii.gz")
    labels_path = os.path.join(results_dir, f"{subject}_lss_conditions.csv")
    if not os.path.exists(beta_path):
        return None, None
    beta_4d    = nib.load(beta_path)
    conditions = pd.read_csv(labels_path).iloc[:, 0].tolist()
    return beta_4d, conditions


def load_lss_betas_test(results_dir, subject):
    beta_4d, conditions = load_lss_betas_full(results_dir, subject)
    if beta_4d is None:
        return None, None
    from nilearn.image import index_img
    test_conds   = [c for c in conditions
                    if parse_condition(c)["phase"].startswith("test")]
    test_indices = [conditions.index(c) for c in test_conds]
    return index_img(beta_4d, test_indices), test_conds


def load_model_rdms(results_dir, tag):
    path = os.path.join(results_dir, f"model_rdms_{tag}.npz")
    if not os.path.exists(path):
        return [], {}
    npz        = np.load(path, allow_pickle=True)
    conditions = ([str(x) for x in npz["conditions"]]
                  if "conditions" in npz.files else [])
    rdms       = {k: npz[k] for k in npz.files if k != "conditions"}
    return conditions, rdms


def rdm_to_vec(rdm):
    return squareform(rdm) if rdm.ndim == 2 else rdm


def extract_sphere_patterns(beta_4d, center_vox, radius_mm):
    data     = beta_4d.get_fdata()
    affine   = beta_4d.affine
    vox_size = np.abs(np.diag(affine)[:3])
    mask     = np.all(data != 0, axis=-1)
    coords   = np.array(np.where(mask)).T
    dists    = np.sqrt(((coords - center_vox)**2 * vox_size**2).sum(1))
    nb       = coords[dists <= radius_mm]
    return data[nb[:, 0], nb[:, 1], nb[:, 2], :].T   # (n_cond, k)


# ==============================================================================
# PLOT 1 -- whole-brain axial slices (masked)
# ==============================================================================

def plot_wholebrain(results_dir, plots_dir, subjects, methods, brain_masks):
    from nilearn import plotting

    for subject in subjects:
        mask_path = brain_masks.get(subject)
        if mask_path and not os.path.exists(mask_path):
            print(f"  WARNING: mask not found for {subject}: {mask_path}")
            mask_path = None

        for prefix, method_label in methods.items():
            maps = {}
            for f in sorted(os.listdir(results_dir)):
                if (f.startswith(f"{subject}_{prefix}_") and
                        f.endswith(".nii.gz")):
                    mname = (f.replace(f"{subject}_{prefix}_", "")
                              .replace(".nii.gz", ""))
                    maps[mname] = os.path.join(results_dir, f)

            if not maps:
                continue

            n   = len(maps)
            fig, axes = plt.subplots(n, 1, figsize=(20, 3.5 * n))
            if n == 1:
                axes = [axes]

            for ax, (mname, nii_path) in zip(axes, maps.items()):
                raw_img  = nib.load(nii_path)

                # Apply brain mask — sets out-of-brain voxels to NaN
                img  = apply_gm_mask(raw_img, mask_path)
                data = img.get_fdata()

                vmax = np.nanpercentile(np.abs(data), 99)
                if vmax == 0 or np.isnan(vmax):
                    ax.set_visible(False)
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plotting.plot_stat_map(
                        img, axes=ax,
                        threshold=0.0,
                        display_mode="z",
                        cut_coords=8,
                        colorbar=True,
                        cmap="RdYlBu_r",
                        vmax=vmax,
                        vmin=-vmax,
                        title=f"{subject} | {mname}",
                    )

            mask_note = "" if mask_path else " (no mask)"
            fig.suptitle(f"{method_label}\n{subject}{mask_note}",
                         fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout()
            out = os.path.join(plots_dir,
                               f"{subject}_{prefix}_wholebrain.png")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")


# ==============================================================================
# PLOT 2 -- generic region neural RDM (raw + ranked + model scatter)
# ==============================================================================

def plot_region_rdm(plots_dir, subject,
                    beta_4d, conditions, model_rdms,
                    region_name, region_mni, radius_mm, out_tag):
    center_vox = mni_to_vox(region_mni, beta_4d.affine)
    patterns   = extract_sphere_patterns(beta_4d, center_vox, radius_mm)
    n_cond     = len(conditions)

    if patterns.shape[0] < n_cond or patterns.shape[1] < 3:
        print(f"  [skip {out_tag}] sphere has only {patterns.shape[1]} "
              f"voxels for {subject}")
        return

    neural_vec = pdist(patterns, metric="correlation")
    neural_rdm = squareform(neural_vec)
    ranked_rdm = squareform(rankdata(neural_vec))
    np.fill_diagonal(ranked_rdm, 0)

    short  = [short_label(c) for c in conditions]
    n_mod  = len(model_rdms)
    n_rows = 1 + n_mod

    fig = plt.figure(figsize=(14, 5 + 5 * n_mod))
    gs  = gridspec.GridSpec(n_rows, 2, hspace=0.5, wspace=0.35)

    # Row 0: raw + ranked neural RDM
    for col, (mat, title, cb_label, vmax) in enumerate([
        (neural_rdm, "Neural RDM (raw 1-r)",  "1-r",
         2.0),
        (ranked_rdm, "Neural RDM (ranked)",    "rank",
         float(np.nanmax(ranked_rdm))),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=vmax,
                       interpolation="none", aspect="auto")
        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(short, fontsize=4, rotation=90)
        ax.set_yticks(range(n_cond))
        ax.set_yticklabels(short, fontsize=4)
        ax.set_title(title, fontweight="bold", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, label=cb_label)

    # Rows 1+: one model per row
    for row, (mname, rdm) in enumerate(model_rdms.items()):
        mvec      = rdm_to_vec(rdm)
        rho, pval = spearmanr(neural_vec, mvec)
        rdm_sq    = squareform(mvec) if mvec.ndim == 1 else rdm
        n_m       = rdm_sq.shape[0]
        short_m   = [short_label(c) for c in conditions[:n_m]]

        ax_m = fig.add_subplot(gs[1 + row, 0])
        im_m = ax_m.imshow(rdm_sq, cmap="RdYlBu_r", vmin=0, vmax=1,
                            interpolation="none", aspect="auto")
        ax_m.set_xticks(range(n_m))
        ax_m.set_xticklabels(short_m, fontsize=4, rotation=90)
        ax_m.set_yticks(range(n_m))
        ax_m.set_yticklabels(short_m, fontsize=4)
        ax_m.set_title(f"Model: {mname}  rho={rho:.3f}  p={pval:.3f}",
                       fontweight="bold", fontsize=9)
        plt.colorbar(im_m, ax=ax_m, fraction=0.046)

        ax_s = fig.add_subplot(gs[1 + row, 1])
        ax_s.scatter(rankdata(mvec), rankdata(neural_vec),
                     alpha=0.3, s=6, color="steelblue", rasterized=True)
        ax_s.set_xlabel(f"Model rank ({mname})", fontsize=8)
        ax_s.set_ylabel("Neural rank", fontsize=8)
        ax_s.set_title(f"Rank scatter  rho={rho:.3f}",
                       fontsize=9, fontweight="bold")

    fig.suptitle(
        f"{subject} | {region_name} {region_mni} | r={radius_mm}mm\n"
        f"n_cond={n_cond}  sphere_voxels={patterns.shape[1]}",
        fontsize=11, fontweight="bold",
    )
    out = os.path.join(plots_dir, f"{subject}_{out_tag}_rdm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ==============================================================================
# PLOT 3 -- searchlight value bar chart at a coordinate
# ==============================================================================

def plot_region_barchart(results_dir, plots_dir, subject, methods,
                          region_name, region_mni, out_tag):
    rows = []
    for prefix in methods:
        for f in sorted(os.listdir(results_dir)):
            if not (f.startswith(f"{subject}_{prefix}_") and
                    f.endswith(".nii.gz")):
                continue
            mname = (f.replace(f"{subject}_{prefix}_", "")
                      .replace(".nii.gz", ""))
            img  = nib.load(os.path.join(results_dir, f))
            cv   = mni_to_vox(region_mni, img.affine)
            data = img.get_fdata()
            val  = (float(data[cv[0], cv[1], cv[2]])
                    if all(0 <= cv[i] < data.shape[i] for i in range(3))
                    else np.nan)
            rows.append({"method": prefix, "model": mname, "value": val})

    if not rows:
        print(f"  [skip barchart {out_tag}] no maps for {subject}")
        return

    df              = pd.DataFrame(rows)
    methods_present = df["method"].unique()
    models_present  = sorted(df["model"].unique())
    n_m             = len(methods_present)

    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4), sharey=False)
    if n_m == 1:
        axes = [axes]

    for ax, pref in zip(axes, methods_present):
        sub  = df[df["method"] == pref].set_index("model")
        vals = [float(sub.loc[m, "value"]) if m in sub.index else np.nan
                for m in models_present]
        colors = ["#d73027" if (v > 0 and not np.isnan(v)) else "#4575b4"
                  for v in vals]
        ax.bar(range(len(models_present)), vals,
               color=colors, edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(len(models_present)))
        ax.set_xticklabels(models_present, rotation=45,
                           ha="right", fontsize=8)
        ax.set_title(methods.get(pref, pref), fontsize=9, fontweight="bold")
        ax.set_ylabel("Searchlight value at voxel")

    fig.suptitle(f"{subject} | {region_name} {region_mni}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{subject}_{out_tag}_barchart.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")




# ==============================================================================
# PLOT 4 -- coordinate-centric ortho slices (3 variants, GM masked)
# ==============================================================================

def _prepare_variants(img):
    """
    Return (img_full, img_pos, img_top10):
      full  -- original, symmetric colorbar
      pos   -- positive values only  (negatives -> NaN)
      top10 -- top 10% of positive values only
    """
    data = img.get_fdata().copy()

    pos_data = data.copy()
    pos_data[pos_data <= 0] = np.nan
    img_pos = nib.Nifti1Image(pos_data, img.affine, img.header)

    pos_vals = data[data > 0]
    if len(pos_vals) > 0:
        thresh   = np.nanpercentile(pos_vals, 90)
        top_data = data.copy()
        top_data[top_data <= thresh] = np.nan
    else:
        top_data = pos_data.copy()
    img_top = nib.Nifti1Image(top_data, img.affine, img.header)

    return img, img_pos, img_top


def plot_region_ortho(results_dir, plots_dir, subject, methods,
                       region_name, region_mni, brain_masks):
    """
    For every searchlight NIfTI map, produce 3 ortho-view PNGs centred on
    region_mni (full / positive-only / top-10-percent), GM-masked.

    Output names:
      {subject}_{prefix}_{model}_{tag}_ortho_full.png
      {subject}_{prefix}_{model}_{tag}_ortho_pos.png
      {subject}_{prefix}_{model}_{tag}_ortho_top10.png
    """
    from nilearn import plotting

    tag       = region_name.lower()
    mask_path = brain_masks.get(subject)
    if (mask_path and mask_path != "gm_prior"
            and not os.path.exists(mask_path)):
        print(f"  WARNING: mask not found for {subject}: {mask_path}")
        mask_path = None

    _ORTHO_VARIANTS = [
        ("full",  "full (pos + neg)"),
        ("pos",   "positive only"),
        ("top10", "top 10% positive"),
    ]

    for prefix in methods:
        nii_files = sorted([
            f for f in os.listdir(results_dir)
            if f.startswith(f"{subject}_{prefix}_") and f.endswith(".nii.gz")
        ])
        if not nii_files:
            continue

        for nii_fname in nii_files:
            model_name = (nii_fname
                          .replace(f"{subject}_{prefix}_", "")
                          .replace(".nii.gz", ""))
            masked_img = apply_gm_mask(
                nib.load(os.path.join(results_dir, nii_fname)), mask_path)

            img_full, img_pos, img_top = _prepare_variants(masked_img)

            for var_tag, var_label in _ORTHO_VARIANTS:
                img_plot = {"full": img_full,
                            "pos":  img_pos,
                            "top10": img_top}[var_tag]

                valid = img_plot.get_fdata()
                valid = valid[~np.isnan(valid)]
                if len(valid) == 0:
                    continue

                vmax     = np.nanpercentile(np.abs(valid), 99)
                has_neg  = np.any(valid < 0)
                cmap     = "RdYlBu_r" if has_neg else "YlOrRd"
                vmin_arg = -vmax if has_neg else 0.0

                if vmax == 0 or np.isnan(vmax):
                    continue

                fig = plt.figure(figsize=(10, 3.5))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    display = plotting.plot_stat_map(
                        img_plot,
                        display_mode = "ortho",
                        cut_coords   = region_mni,
                        threshold    = 0.0,
                        colorbar     = True,
                        cmap         = cmap,
                        vmax         = vmax,
                        vmin         = vmin_arg,
                        figure       = fig,
                        title        = (f"{subject} | {model_name} | "
                                        f"{region_name} {region_mni} | "
                                        f"{var_label}"),
                    )
                    display.add_markers(
                        [region_mni],
                        marker_color = "lime",
                        marker_size  = 80,
                    )

                out = os.path.join(
                    plots_dir,
                    f"{subject}_{prefix}_{model_name}_{tag}_ortho_{var_tag}.png"
                )
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {out}")

# ==============================================================================
# MAIN
# ==============================================================================

REGIONS = [
    # (display name,  MNI coord,  out_tag prefix)
    ("DLPFC",         DLPFC_MNI,  "dlpfc"),
    ("LOC",           LOC_MNI,    "loc"),
    ("M1",            ( -38, -22,  56), "m1"),   # left primary motor cortex (hand area)
    ("SMA",          (   0, -10,  60), "sma"),  # supplementary motor area
]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Results dir : {RESULTS_DIR}")
    print(f"Plots dir   : {PLOTS_DIR}")
    print(f"Subjects    : {SUBJECTS}")
    print(f"DLPFC MNI   : {DLPFC_MNI}")
    print(f"LOC MNI     : {LOC_MNI}  (left Lateral Occipital Complex)")
    print()

    # 1. Whole-brain masked stat maps
    print("=" * 55)
    # print("Whole-brain stat maps (masked)")
    # print("=" * 55)
    # plot_wholebrain(RESULTS_DIR, PLOTS_DIR, SUBJECTS, METHODS, BRAIN_MASKS)

    # Load model RDMs once
    _, model_rdms_full = load_model_rdms(RESULTS_DIR, "full")
    _, model_rdms_test = load_model_rdms(RESULTS_DIR, "test")

    if not model_rdms_full:
        print("WARNING: model_rdms_full.npz not found")
    if not model_rdms_test:
        print("WARNING: model_rdms_test.npz not found")

    # 2+3. Region RDMs and bar charts
    for subject in SUBJECTS:
        print(f"\n{'='*55}\n  {subject}\n{'='*55}")

        beta_full, conds_full = load_lss_betas_full(RESULTS_DIR, subject)
        beta_test, conds_test = load_lss_betas_test(RESULTS_DIR, subject)

        if beta_full is None:
            print(f"  LSS betas not found for {subject} -- skipping")
            continue

        for region_name, region_mni, tag in REGIONS:
            print(f"\n  -- {region_name} {region_mni} --")

            if model_rdms_test and beta_test is not None:
                print(f"  {region_name}: test-phase RDM ...")
                plot_region_rdm(
                    PLOTS_DIR, subject,
                    beta_test, conds_test, model_rdms_test,
                    region_name=region_name,
                    region_mni=region_mni,
                    radius_mm=RADIUS_MM,
                    out_tag=f"{tag}_test",
                )

            if model_rdms_full:
                print(f"  {region_name}: full-condition RDM ...")
                plot_region_rdm(
                    PLOTS_DIR, subject,
                    beta_full, conds_full, model_rdms_full,
                    region_name=region_name,
                    region_mni=region_mni,
                    radius_mm=RADIUS_MM,
                    out_tag=f"{tag}_full",
                )

            print(f"  {region_name}: bar chart ...")
            plot_region_barchart(
                RESULTS_DIR, PLOTS_DIR, subject, METHODS,
                region_name=region_name,
                region_mni=region_mni,
                out_tag=tag,
            )

            print(f"  {region_name}: ortho slices (3 variants) ...")
            plot_region_ortho(
                RESULTS_DIR, PLOTS_DIR, subject, METHODS,
                region_name=region_name,
                region_mni=region_mni,
                brain_masks=BRAIN_MASKS,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()