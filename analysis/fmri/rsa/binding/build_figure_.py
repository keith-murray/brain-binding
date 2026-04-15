"""
build_figure.py  — individual panel PNGs, same total width per row.

Saves (all to OUT_DIR):
  row_A.png                 — 4 model RDMs + shared colorbar
  row_B_loc_rdm.png         — LOC neural RDM + colorbar
  row_B_loc_scatter.png     — LOC rank scatter
  row_B_sma_rdm.png         — SMA neural RDM + colorbar
  row_B_sma_scatter.png     — SMA rank scatter
  row_C_dlpfc_rdm.png       — DLPFC neural RDM + colorbar
  row_C_dlpfc_scatter.png   — DLPFC rank scatter (rule type)
  row_C_ortho.png           — DLPFC ortho map
  row_D.png                 — 3 coefficient panels

Panel widths are proportioned so every row sums to FIG_W = 7.09 in.
No other logic is changed.
"""

import os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr, linregress

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR     = "/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding/rsa_results_masked"
OUT_DIR         = os.path.join(RESULTS_DIR, "plots_final")

FOCAL_SUBJECT   = "sub-008"
ALL_SUBJECTS    = ["sub-001", "sub-002", "sub-008"]
SUBJECT_LABELS  = {"sub-001": "sub-001", "sub-002": "sub-002", "sub-008": "sub-003"}

LOC_MNI   = (-42, -74, -16)
SMA_MNI   = (  0,  -4,  56)
DLPFC_MNI = (-44,  36,  28)
RADIUS_MM = 12.0

GM_MASK_PATH    = "/jukebox/PNI-classes/students/NEU502/2026-NEU502B/binary_shaef.nii"
SPEARMAN_PREFIX = "spearman_full"

# ══════════════════════════════════════════════════════════════════════════════
# PANEL DIMENSIONS  — all rows sum to FIG_W
# ══════════════════════════════════════════════════════════════════════════════

FIG_W = 7.09   # inches  (Nature 180 mm)
DPI   = 300

# Row A: 4 equal RDMs + 1 colorbar
_A_CBAR_W = 0.18
_A_RDM_W  = (FIG_W - _A_CBAR_W) / 4      # ≈ 1.73
ROW_A_H   = 1.75

# Row B: ratios [1.1, 0.9, 1.1, 0.9]  →  [1.95, 1.60, 1.95, 1.60]
_B_RATIOS = [1.1, 0.9, 1.1, 0.9]
_B_TOTAL  = sum(_B_RATIOS)
B_LOC_RDM_W, B_LOC_SCAT_W, B_SMA_RDM_W, B_SMA_SCAT_W = [
    FIG_W * r / _B_TOTAL for r in _B_RATIOS]
ROW_B_H   = 1.85

# Row C: 
_C_RATIOS = [1.1, 0.9, 2]
_C_TOTAL  = sum(_C_RATIOS)
C_RDM_W, C_SCAT_W, C_ORTHO_W = [FIG_W * r / _C_TOTAL for r in _C_RATIOS]
ROW_C_H   = 1.85

# Row D: full width
ROW_D_H   = 1.90

RDM_CMAP = "RdYlBu_r"

# ══════════════════════════════════════════════════════════════════════════════
# STYLE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":          7,
    "axes.labelsize":     7,
    "axes.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.0,
    "ytick.major.size":   2.0,
    "lines.linewidth":    0.8,
    "pdf.fonttype":       42,
})

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (unchanged from document)
# ══════════════════════════════════════════════════════════════════════════════

def parse_condition(label):
    parts = str(label).split("_")
    rule_type = parts[3] if len(parts) == 4 else None
    return {"shape": parts[0], "role": parts[1], "phase": parts[2],
            "rule_type": rule_type,
            "epoch": "rule" if parts[2].startswith("rule") else "test"}

_SHAPE_SHORT = {"circle":"Cir","rectangle":"Rec","star":"Sta","triangle":"Tri"}
_ROLE_SYM    = {"A":"A","B":"B","Aprime":"A'","Bprime":"B'"}
_PHASE_SHORT = {"rule1":"r1","rule2":"r2","rule3":"r3","test1":"t1","test2":"t2"}

def short_label(cond):
    p = parse_condition(cond)
    return (f"{_SHAPE_SHORT.get(p['shape'], p['shape'][:3])}\n"
            f"{_ROLE_SYM.get(p['role'], p['role'])}\n"
            f"{_PHASE_SHORT.get(p['phase'], p['phase'])}"
            f"{'_'+p['rule_type'] if p['rule_type'] else ''}")

def mni_to_vox(mni_coord, affine):
    inv = np.linalg.inv(affine)
    return np.round(inv[:3,:3] @ np.array(mni_coord) + inv[:3,3]).astype(int)

def load_betas(subject):
    b4d  = nib.load(os.path.join(RESULTS_DIR, f"{subject}_lss_beta_4d.nii.gz"))
    cond = pd.read_csv(os.path.join(RESULTS_DIR,
                       f"{subject}_lss_conditions.csv")).iloc[:,0].tolist()
    return b4d, cond

def load_model_rdms(tag="full"):
    npz   = np.load(os.path.join(RESULTS_DIR, f"model_rdms_{tag}.npz"),
                    allow_pickle=True)
    conds = [str(x) for x in npz["conditions"]]
    rdms  = {k: npz[k] for k in npz.files if k != "conditions"}
    return conds, rdms

def extract_sphere(beta_4d, center_vox, radius_mm):
    data     = beta_4d.get_fdata()
    vox_size = np.abs(np.diag(beta_4d.affine)[:3])
    mask     = np.all(data != 0, axis=-1)
    coords   = np.array(np.where(mask)).T
    dists    = np.sqrt(((coords - center_vox)**2 * vox_size**2).sum(1))
    nb       = coords[dists <= radius_mm]
    return data[nb[:,0], nb[:,1], nb[:,2], :].T   # (n_cond, k)

def apply_gm_mask(stat_img, threshold=0.4):
    from nilearn.image import resample_to_img, math_img
    if os.path.exists(GM_MASK_PATH):
        gm_raw = nib.load(GM_MASK_PATH)
        gm_img = nib.Nifti1Image(
            (gm_raw.get_fdata() > 0).astype(np.uint8), gm_raw.affine)
    else:
        from nilearn.datasets import fetch_icbm152_2009
        mni    = fetch_icbm152_2009()
        gm_img = math_img(f"img > {threshold}", img=nib.load(mni["gm"]))
    gm_res = resample_to_img(gm_img, stat_img, interpolation="nearest")
    d      = stat_img.get_fdata().copy()
    d[~gm_res.get_fdata().astype(bool)] = np.nan
    return nib.Nifti1Image(d, stat_img.affine, stat_img.header)

def rdm_to_vec(rdm):
    return squareform(rdm) if rdm.ndim == 2 else rdm

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def draw_rdm(ax, rdm_matrix, vmin=0, vmax=1.5, cmap=RDM_CMAP):
    im = ax.imshow(rdm_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="none", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    return im

def _attach_cbar(fig, ax, im, label=None):
    """Attach a slim colorbar to the right of ax."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)
    if label:
        cbar.set_label(label, fontsize=6, labelpad=2)
    return cbar

def draw_box_overlay(ax, row_indices, col_indices, color, lw=1.0):
    r0, r1 = min(row_indices)-.5, max(row_indices)+.5
    c0, c1 = min(col_indices)-.5, max(col_indices)+.5
    ax.add_patch(mpatches.Rectangle(
        (c0, r0), c1-c0, r1-r0,
        linewidth=lw, edgecolor=color, facecolor="none", clip_on=True))

def draw_scatter_rank(ax, neural_vec, model_vec, model_name, color="steelblue"):
    # Guard: constant model vector
    if np.nanstd(model_vec) < 1e-10:
        ax.text(0.5, 0.5, "model constant", ha="center", va="center",
                transform=ax.transAxes, fontsize=6, color="gray")
        ax.set_xlabel(f"{model_name} rank", fontsize=6.5, labelpad=2)
        ax.set_ylabel("Neural rank", fontsize=6.5, labelpad=2)
        return
    rx  = rankdata(model_vec)
    rx = rx/rx.max()
    ry  = rankdata(neural_vec)
    rho, pval = spearmanr(neural_vec, model_vec)

    ax.scatter(rx, ry, s=2, alpha=0.25, color=color,
               linewidths=0, rasterized=True)
    if np.std(rx) > 1e-10:
        m, b, *_ = linregress(rx, ry)
        xl = np.array([rx.min(), rx.max()])
        ax.plot(xl, m*xl + b, color=color, lw=0.9)
    p_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.text(0.97, 0.05, f"ρ={rho:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="0.7", lw=0.4, alpha=0.9))
    ax.set_xticks([rx.min(), 1])
    ax.set_xticklabels(["low", "high"])
    ax.set_xlabel(f"{model_name} dissimilarity", fontsize=6.5, labelpad=2)
    ax.set_ylabel("Neural rank", fontsize=6.5, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_xlim(rx.min()-0.1, rx.max()+0.1)

def draw_ortho(ax, nii_img, cut_coords, threshold=0.1):
    from nilearn import plotting
    fig_tmp = plt.figure(figsize=(8.0, 2.5), dpi=150)
    ax_tmp  = fig_tmp.add_axes([0, 0, 1, 1])
    d    = nii_img.get_fdata()
    pos  = d[d > 0]
    vmax = float(np.nanpercentile(pos, 98)) if len(pos) else 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = plotting.plot_stat_map(
            nii_img, display_mode="ortho", cut_coords=cut_coords,
            threshold=threshold, vmax=vmax, vmin=0,
            cmap="YlOrRd", colorbar=True,
            axes=ax_tmp, figure=fig_tmp, annotate=False)
        disp.add_markers([cut_coords], marker_color="lime", marker_size=40)
    fig_tmp.canvas.draw()
    buf     = np.frombuffer(fig_tmp.canvas.buffer_rgba(), dtype=np.uint8)
    w, h    = fig_tmp.canvas.get_width_height()
    img_arr = buf.reshape(h, w, 4)[:, :, :3]
    plt.close(fig_tmp)
    ax.imshow(img_arr, aspect="equal", interpolation="lanczos")
    ax.axis("off")

def _save(fig, fname):
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD — one PNG per panel / logical group
# ══════════════════════════════════════════════════════════════════════════════

def build_figure():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading data ...")
    beta_4d, conditions = load_betas(FOCAL_SUBJECT)
    _, model_rdms       = load_model_rdms("full")
    n_cond = len(conditions)
    print(f"  {FOCAL_SUBJECT}: {n_cond} conds, shape {beta_4d.shape}")

    # ── Pre-compute region patterns & neural RDMs once ──────────────────────
    def sphere_rdm(mni):
        cv   = mni_to_vox(mni, beta_4d.affine)
        patt = extract_sphere(beta_4d, cv, RADIUS_MM)
        vec  = pdist(patt, metric="correlation")
        return patt, vec, squareform(vec)

    patt_loc, vec_loc, rdm_loc   = sphere_rdm(LOC_MNI)
    patt_sma, vec_sma, rdm_sma   = sphere_rdm(SMA_MNI)
    patt_dlp, vec_dlp, rdm_dlp   = sphere_rdm(DLPFC_MNI)

    m_shape = rdm_to_vec(model_rdms.get("M1_shape",
                          np.zeros((n_cond, n_cond))))
    m_epoch = rdm_to_vec(model_rdms.get("M4_epoch",
                          np.zeros((n_cond, n_cond))))
    m_rule  = rdm_to_vec(model_rdms.get("M5_rule_type",
                          np.zeros((n_cond, n_cond))))

    # ── ROW A: 4 model RDMs + shared colorbar  (full width) ─────────────────
    print("Row A ...")
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, ROW_A_H))
    fig.subplots_adjust(wspace=0.18, left=0.02, right=0.88,
                        top=0.90, bottom=0.05)

    KEYS_A   = ["M1_shape", "M4_epoch", "M5_rule_type", "M2_abstract_role"]
    TITLES_A = ["shape", "epoch", "rule type", "abstract role"]
    last_im  = None
    for ax, key, title in zip(axes, KEYS_A, TITLES_A):
        rdm = model_rdms.get(key)
        if rdm is None:
            ax.set_visible(False); continue
        rdm_sq = squareform(rdm) if rdm.ndim == 1 else rdm
        im     = draw_rdm(ax, rdm_sq, vmin=0, vmax=1)
        ax.set_title(title, fontsize=7, pad=2)
        last_im = im

    # shared colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("dissimilarity", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)

    _save(fig, "row_A.png")

    # ── ROW B — LOC RDM ──────────────────────────────────────────────────────
    print("Row B ...")
    fig, ax = plt.subplots(1, 1, figsize=(B_LOC_RDM_W, ROW_B_H))
    fig.subplots_adjust(left=0.05, right=0.80, top=0.88, bottom=0.05)
    im = draw_rdm(ax, rdm_loc, vmin=0, vmax=2)
    shape_colors = {"circle":"#e41a1c","rectangle":"#377eb8",
                    "star":"#4daf4a","triangle":"#984ea3"}
    for shape, col in shape_colors.items():
        idx = [i for i, c in enumerate(conditions)
               if parse_condition(c)["shape"] == shape]
        if idx: draw_box_overlay(ax, idx, idx, col)
    ax.set_title(f"LOC  {LOC_MNI}", fontsize=7, pad=2)
    _attach_cbar(fig, ax, im, label="1 − r")
    _save(fig, "row_B_loc_rdm.png")

    # ── ROW B — LOC scatter ───────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(B_LOC_SCAT_W, ROW_B_H))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.18)
    draw_scatter_rank(ax, vec_loc, m_shape, "shape", "#e41a1c")
    _save(fig, "row_B_loc_scatter.png")

    # ── ROW B — SMA RDM ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(B_SMA_RDM_W, ROW_B_H))
    fig.subplots_adjust(left=0.05, right=0.80, top=0.88, bottom=0.05)
    im = draw_rdm(ax, rdm_sma, vmin=0, vmax=2)
    rule_idx = [i for i, c in enumerate(conditions)
                if parse_condition(c)["epoch"] == "rule"]
    test_idx = [i for i, c in enumerate(conditions)
                if parse_condition(c)["epoch"] == "test"]
    # if rule_idx: draw_box_overlay(ax, rule_idx, rule_idx, "#ff7f00")
    # if test_idx: draw_box_overlay(ax, test_idx, test_idx, "#a65628")
    ax.set_title(f"SMA  {SMA_MNI}", fontsize=7, pad=2)
    _attach_cbar(fig, ax, im, label="1 − r")
    _save(fig, "row_B_sma_rdm.png")

    # ── ROW B — SMA scatter ───────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(B_SMA_SCAT_W, ROW_B_H))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.18)
    # epoch is degenerate in test-phase; use rule_type for SMA scatter
    draw_scatter_rank(ax, vec_sma, m_epoch, "epoch type", "#ff7f00")
    _save(fig, "row_B_sma_scatter.png")

    # ── ROW C — DLPFC RDM ────────────────────────────────────────────────────
    print("Row C ...")
    fig, ax = plt.subplots(1, 1, figsize=(C_RDM_W, ROW_C_H))
    fig.subplots_adjust(left=0.05, right=0.80, top=0.88, bottom=0.05)
    im = draw_rdm(ax, rdm_dlp, vmin=0, vmax=1.5)
    ax.set_title(f"DLPFC  {DLPFC_MNI}", fontsize=7, pad=2)
    # rho inset (informative pairs only, 0.5 masked)
    mask       = m_rule != 0.5
    rho_v, p_v = spearmanr(vec_dlp[mask], m_rule[mask])
    p_str      = "p<0.001" if p_v < 0.001 else f"p={p_v:.3f}"
    ax.text(0.97, 0.03, f"ρ={rho_v:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=5.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="0.7", lw=0.4, alpha=0.9))
    _attach_cbar(fig, ax, im, label="1 − r")
    _save(fig, "row_C_dlpfc_rdm.png")

    # ── ROW C — DLPFC scatter ─────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(C_SCAT_W, ROW_C_H))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.92, bottom=0.20)
    draw_scatter_rank(ax, vec_dlp[mask], m_rule[mask],
                      "rule type", "#1b7837")
    _save(fig, "row_C_dlpfc_scatter.png")

    # ── ROW C — ortho ─────────────────────────────────────────────────────────
    rho_path = os.path.join(RESULTS_DIR,
                             f"{FOCAL_SUBJECT}_{SPEARMAN_PREFIX}_M5_rule_type.nii.gz")
    if not os.path.exists(rho_path):
        rho_path = os.path.join(RESULTS_DIR,
                                 f"{FOCAL_SUBJECT}_spearman_test_rule_type_test.nii.gz")
    fig, ax = plt.subplots(1, 1, figsize=(C_ORTHO_W, ROW_C_H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if os.path.exists(rho_path):
        masked = apply_gm_mask(nib.load(rho_path))
        d      = masked.get_fdata().copy()
        d[d <= 0] = np.nan
        draw_ortho(ax, nib.Nifti1Image(d, masked.affine, masked.header),
                   DLPFC_MNI, threshold=0.1)
    else:
        ax.text(0.5, 0.5, "searchlight map not found",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="gray")
        ax.axis("off")
    _save(fig, "row_C_ortho.png")

    # ── ROW D: 3 coefficient panels (full width) ─────────────────────────────
    print("Row D ...")
    try:
        _, model_rdms_test = load_model_rdms("test")
    except Exception:
        model_rdms_test = {}

    FULL_KEYS   = ["M1_shape", "M4_epoch", "M5_rule_type", "M2_abstract_role"]
    FULL_LABELS = ["shape\n(full)", "epoch", "rule type", "abstract role"]
    ALL_KEYS    = FULL_KEYS
    ALL_LABELS  = FULL_LABELS
    n_models    = len(ALL_KEYS)

    subject_colors = {"sub-001":"#1a9850","sub-002":"#4393c3","sub-008":"#d6604d"}
    dot_offset     = np.linspace(-0.15, 0.15, len(ALL_SUBJECTS))

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, ROW_D_H))
    fig.subplots_adjust(wspace=0.40, left=0.07, right=0.97,
                        top=0.88, bottom=0.22)

    for ax, (region_name, region_mni) in zip(
            axes, [("LOC", LOC_MNI), ("SMA", SMA_MNI), ("DLPFC", DLPFC_MNI)]):

        vals = {s: [] for s in ALL_SUBJECTS}
        for subject in ALL_SUBJECTS:
            b4d, subj_conds = load_betas(subject)
            cv              = mni_to_vox(region_mni, b4d.affine)
            patt            = extract_sphere(b4d, cv, RADIUS_MM)
            vec_full        = pdist(patt, metric="correlation")

            for key in ALL_KEYS:
                rdm_m = model_rdms.get(key)
                if rdm_m is None:
                    vals[subject].append(np.nan); continue
                rdm_sq = squareform(rdm_m) if rdm_m.ndim == 1 else rdm_m
                mvec   = squareform(rdm_sq)
                if np.nanstd(mvec) < 1e-10:
                    vals[subject].append(np.nan)
                else:
                    rho, _ = spearmanr(vec_full, mvec)
                    vals[subject].append(rho)

        all_vals = np.array([vals[s] for s in ALL_SUBJECTS])
        mean_v   = np.nanmean(all_vals, axis=0)
        bar_cols = ["#d73027" if (not np.isnan(v) and v > 0) else "#4575b4"
                    for v in mean_v]
        x = np.arange(n_models)
        ax.bar(x, np.where(np.isnan(mean_v), 0, mean_v),
               color=bar_cols, edgecolor="k",
               linewidth=0.4, width=0.55, alpha=0.65, zorder=2)
        ax.axhline(0, color="k", linewidth=0.4, zorder=1)
        for s_idx, subject in enumerate(ALL_SUBJECTS):
            ax.scatter(x + dot_offset[s_idx], vals[subject],
                       s=14, color=subject_colors.get(subject, "gray"),
                       zorder=3, linewidths=0.3, edgecolors="white",
                       label=SUBJECT_LABELS.get(subject, subject))
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_LABELS, rotation=35, ha="right", fontsize=6)
        ax.set_yticks([])
        ax.set_ylabel("Spearman ρ", fontsize=6.5, labelpad=2)
        ax.set_title(region_name, fontsize=7, pad=2)
        if region_name == "DLPFC":
            ax.legend(fontsize=5.5, frameon=False,
                      loc="upper left", handletextpad=0.2)

    _save(fig, "row_D.png")
    print("\nAll panels saved to", OUT_DIR)


if __name__ == "__main__":
    build_figure()