"""
build_figure.py
-------------
Composite RSA figure in Nature style.

Layout
------
Row A : Model RDMs          (A1 shape  A2 epoch  A3 rule_type  A4 abstract_role)
Row B : Sanity-check RDMs   (B1 LOC-RDM  B2 LOC-scatter  B3 SMA-RDM  B4 SMA-scatter)
Row C : Main finding        (C1 DLPFC-RDM  C2 DLPFC ortho map)
Row D : Coefficient panels  (D1 LOC  D2 SMA  D3 DLPFC)

Design notes
------------
* Nature full-width: 180 mm / 7.09 in
* Fonts: 7 pt tick labels, 8 pt axis labels (Arial via rcParams)
* No figure titles — panels stitched externally with TikZ
* Model RDMs: x-ticks only (condition labels), no y-ticks
* Neural RDMs: raw correlation distance, no ticks (clean)
* Scatter plots: rank space, Spearman rho + fit line as inset
* sub-008 labelled "sub-003" per request
* Ortho map thresholded at rho > 0.1, centred on DLPFC
* GM mask applied to ortho map
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr, linregress

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit paths
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding/rsa_results_masked"
OUT_PATH    = os.path.join(RESULTS_DIR, "plots_final", "figure_rsa.pdf")

# Subject used for neural RDM panels (B, C) — labelled "sub-003" in figure
FOCAL_SUBJECT   = "sub-008"
SUBJECT_LABEL   = "sub-003"

# All subjects for coefficient panel D
ALL_SUBJECTS    = ["sub-001", "sub-002", "sub-008"]       # add sub-001 when available
SUBJECT_LABELS  = {"sub-001": "sub-001",
                   "sub-002": "sub-002",
                   "sub-008": "sub-003"}

# Region MNI coordinates
LOC_MNI   = (-42, -74, -16)   # left LOC (shifted slightly inferior)
SMA_MNI   = (  0,  -4,  56)   # SMA / premotor — epoch/response encoding
DLPFC_MNI = (-44,  36,  28)   # left DLPFC — rule encoding

RADIUS_MM = 12.0

# GM mask (binary_shaef or "gm_prior")
GM_MASK_PATH = "/jukebox/PNI-classes/students/NEU502/2026-NEU502B/binary_shaef.nii"

# Searchlight prefix used for coefficient panel D
SPEARMAN_PREFIX = "spearman_full"   # one rho per model per voxel

# ══════════════════════════════════════════════════════════════════════════════
# NATURE STYLE
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          7,
    "axes.labelsize":     8,
    "axes.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    0.8,
    "pdf.fonttype":       42,   # embed fonts
    "svg.fonttype":       "none",
})

FIG_W   = 7.09   # inches (180 mm)
FIG_H   = 8.20
RDM_CMAP = "RdYlBu_r"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_condition(label):
    parts = str(label).split("_")
    shape, role, phase = parts[0], parts[1], parts[2]
    rule_type = parts[3] if len(parts) == 4 else None
    return {"shape": shape, "role": role, "phase": phase,
            "rule_type": rule_type,
            "epoch": "rule" if phase.startswith("rule") else "test"}


_SHAPE_SHORT = {"circle": "Cir", "rectangle": "Rec",
                "star": "Sta",   "triangle": "Tri"}
_ROLE_SYM    = {"A": "A", "B": "B", "Aprime": "A'", "Bprime": "B'"}
_PHASE_SHORT = {"rule1": "r1", "rule2": "r2", "rule3": "r3",
                "test1": "t1", "test2": "t2"}


def short_label(cond):
    p     = parse_condition(cond)
    shape = _SHAPE_SHORT.get(p["shape"], p["shape"][:3])
    role  = _ROLE_SYM.get(p["role"], p["role"])
    phase = _PHASE_SHORT.get(p["phase"], p["phase"])
    rule  = f"_{p['rule_type']}" if p["rule_type"] else ""
    return f"{shape}\n{role}\n{phase}{rule}"


def mni_to_vox(mni_coord, affine):
    inv = np.linalg.inv(affine)
    return np.round(inv[:3, :3] @ np.array(mni_coord) + inv[:3, 3]).astype(int)


def load_betas(subject):
    beta_path   = os.path.join(RESULTS_DIR, f"{subject}_lss_beta_4d.nii.gz")
    labels_path = os.path.join(RESULTS_DIR, f"{subject}_lss_conditions.csv")
    beta_4d     = nib.load(beta_path)
    conditions  = pd.read_csv(labels_path).iloc[:, 0].tolist()
    return beta_4d, conditions


def load_model_rdms(tag="full"):
    npz  = np.load(os.path.join(RESULTS_DIR, f"model_rdms_{tag}.npz"),
                   allow_pickle=True)
    conds = [str(x) for x in npz["conditions"]]
    rdms  = {k: npz[k] for k in npz.files if k != "conditions"}
    return conds, rdms


def extract_sphere(beta_4d, center_vox, radius_mm):
    data     = beta_4d.get_fdata()
    affine   = beta_4d.affine
    vox_size = np.abs(np.diag(affine)[:3])
    mask     = np.all(data != 0, axis=-1)
    coords   = np.array(np.where(mask)).T
    dists    = np.sqrt(((coords - center_vox)**2 * vox_size**2).sum(1))
    nb       = coords[dists <= radius_mm]
    return data[nb[:, 0], nb[:, 1], nb[:, 2], :].T   # (n_cond, k)


def apply_gm_mask(stat_img, threshold=0.4):
    from nilearn.image import resample_to_img, math_img
    if os.path.exists(GM_MASK_PATH):
        gm_raw  = nib.load(GM_MASK_PATH)
        gm_data = (gm_raw.get_fdata() > 0).astype(np.uint8)
        gm_img  = nib.Nifti1Image(gm_data, gm_raw.affine)
    else:
        from nilearn.datasets import fetch_icbm152_2009
        mni    = fetch_icbm152_2009()
        gm_img = math_img(f"img > {threshold}", img=nib.load(mni["gm"]))
    gm_res   = resample_to_img(gm_img, stat_img, interpolation="nearest")
    gm_mask  = gm_res.get_fdata().astype(bool)
    stat_d   = stat_img.get_fdata().copy()
    stat_d[~gm_mask] = np.nan
    return nib.Nifti1Image(stat_d, stat_img.affine, stat_img.header)


def rdm_to_vec(rdm):
    return squareform(rdm) if rdm.ndim == 2 else rdm


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def draw_rdm(ax, rdm_matrix, conditions, xticks=False, yticks=False,
             vmin=0, vmax=1, cmap=RDM_CMAP, cbar=False):
    """Draw an RDM heatmap. x-tick labels only if xticks=True."""
    n = rdm_matrix.shape[0]
    im = ax.imshow(rdm_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="none", aspect="equal")
    if xticks and conditions is not None:
        labels = [short_label(c) for c in conditions]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=5, rotation=90, ha="center")
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    return im


def add_cbar_right(
    fig,
    ax,
    *,
    mappable=None,
    cmap=None,
    vmin=None,
    vmax=None,
    label=None,
    pad=0.006,
    width=0.010,
    ticksize=6,
):
    """Add a vertical colorbar immediately to the right of an axis."""

    if mappable is None:
        if cmap is None or vmin is None or vmax is None:
            raise ValueError("Provide either mappable, or (cmap, vmin, vmax)")
        sm = matplotlib.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        mappable = sm

    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + pad, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=ticksize, width=0.4, length=2)
    if label is not None:
        cbar.set_label(label, fontsize=7, labelpad=3)
    return cbar


def add_cbar_in_axis(
    fig,
    cax,
    *,
    mappable=None,
    cmap=None,
    vmin=None,
    vmax=None,
    label=None,
    ticksize=6,
):
    """Add a vertical colorbar in an existing axis (so it participates in GridSpec)."""

    if mappable is None:
        if cmap is None or vmin is None or vmax is None:
            raise ValueError("Provide either mappable, or (cmap, vmin, vmax)")
        sm = matplotlib.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        mappable = sm

    cbar = fig.colorbar(mappable, cax=cax)
    # cbar.ax.tick_params(labelsize=ticksize, width=0.4, length=2)
    # if label is not None:
        # cbar.set_label(label, fontsize=7, labelpad=3)

    # Make the colorbar axis look like a subplot (no extra frame)
    # for spine in cax.spines.values():
        # spine.set_linewidth(0.3)

    return cbar


def draw_box_overlay(ax, row_indices, col_indices, color, lw=1.0):
    """
    Draw a rectangle overlay on an RDM covering the block defined by
    row_indices and col_indices.
    """
    r0, r1 = min(row_indices)-0.5, max(row_indices)+0.5
    c0, c1 = min(col_indices)-0.5, max(col_indices)+0.5
    rect = mpatches.Rectangle(
        (c0, r0), c1-c0, r1-r0,
        linewidth=lw, edgecolor=color, facecolor="none",
        clip_on=True,
    )
    ax.add_patch(rect)


def draw_scatter_rank(ax, neural_vec, model_vec, model_name, color="steelblue"):
    """
    Rank-space scatter plot of neural vs model RDM distances.
    Shows Spearman rho as text and a fitted line.
    """
    rx = rankdata(model_vec)
    ry = rankdata(neural_vec)
    rho, pval = spearmanr(neural_vec, model_vec)

    ax.scatter(rx, ry, s=2, alpha=0.25, color=color,
               linewidths=0, rasterized=True)

    # Fitted line in rank space
    slope, intercept, *_ = linregress(rx, ry)
    x_line = np.array([rx.min(), rx.max()])
    ax.plot(x_line, slope*x_line + intercept,
            color=color, lw=0.8, alpha=0.9)

    # rho inset
    p_str = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.text(0.97, 0.05,
            f"ρ={rho:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=6,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="0.7", lw=0.4, alpha=0.9))

    ax.set_xlabel(f"{model_name} rank", fontsize=7, labelpad=2)
    ax.set_ylabel("Neural rank", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_xlim(rx.min()-1, rx.max()+1)

# ── 2. Canvas buffer: tostring_rgb removed in newer matplotlib ───────────────
def draw_ortho(ax, nii_img, cut_coords, threshold=0.1):
    from nilearn import plotting

    fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(6, 1.8))

    data  = nii_img.get_fdata()
    valid = data[~np.isnan(data)]
    pos   = valid[valid > 0]
    vmax  = np.nanpercentile(pos, 98) if len(pos) > 0 else 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_stat_map(
            nii_img,
            display_mode = "ortho",
            cut_coords   = cut_coords,
            threshold    = threshold,
            vmax         = vmax,
            vmin         = 0,
            cmap         = "YlOrRd",
            colorbar     = True,
            axes         = ax_tmp,
            figure       = fig_tmp,
            annotate     = False,
        )
        display.add_markers(
            [cut_coords],
            marker_color = "lime",
            marker_size  = 40,
        )

    fig_tmp.canvas.draw()

    # tostring_rgb removed in matplotlib >= 3.8 — use buffer_rgba instead
    buf      = np.frombuffer(fig_tmp.canvas.buffer_rgba(), dtype=np.uint8)
    w, h     = fig_tmp.canvas.get_width_height()
    img_arr  = buf.reshape(h, w, 4)[:, :, :3]   # RGBA -> RGB

    plt.close(fig_tmp)

    ax.imshow(img_arr, aspect="auto", interpolation="lanczos")
    ax.axis("off")
    # NOTE: avoid plt.tight_layout() here; it can disturb the main GridSpec layout.

def draw_ortho(ax, nii_img, cut_coords, threshold=0.1):
    from nilearn import plotting

    # Match natural ortho aspect: 3 panels wide, rendered at high res
    fig_tmp = plt.figure(figsize=(8.0, 2.5), dpi=150)
    ax_tmp  = fig_tmp.add_axes([0, 0, 1, 1])   # fill entire figure, no margins

    data  = nii_img.get_fdata()
    valid = data[~np.isnan(data)]
    pos   = valid[valid > 0]
    vmax  = np.nanpercentile(pos, 98) if len(pos) > 0 else 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_stat_map(
            nii_img,
            display_mode = "ortho",
            cut_coords   = cut_coords,
            threshold    = threshold,
            vmax         = vmax,
            vmin         = 0,
            cmap         = "YlOrRd",
            colorbar     = True,
            axes         = ax_tmp,
            figure       = fig_tmp,
            annotate     = False,
        )
        display.add_markers(
            [cut_coords],
            marker_color = "lime",
            marker_size  = 40,
        )

    fig_tmp.canvas.draw()
    buf     = np.frombuffer(fig_tmp.canvas.buffer_rgba(), dtype=np.uint8)
    w, h    = fig_tmp.canvas.get_width_height()
    img_arr = buf.reshape(h, w, 4)[:, :, :3]
    plt.close(fig_tmp)

    ax.imshow(img_arr, aspect="equal", interpolation="lanczos")
    ax.axis("off")
# ══════════════════════════════════════════════════════════════════════════════
# PER-ROW DRAWING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_row_A(axes, model_rdms, conditions):
    """
    A1: shape  A2: epoch  A3: rule_type  A4: abstract_role
    x-ticks on all; no y-ticks; no colorbars (saves space)
    """
    model_keys = ["M1_shape", "M4_epoch", "M5_rule_type", "M2_abstract_role"]
    panel_labels = ["shape", "epoch", "rule type", "abstract role"]

    for ax, key, label in zip(axes, model_keys, panel_labels):
        if key not in model_rdms:
            ax.set_visible(False)
            continue
        rdm = model_rdms[key]
        rdm_sq = squareform(rdm) if rdm.ndim == 1 else rdm
        draw_rdm(ax, rdm_sq, conditions, xticks=False, yticks=False,
             vmin=0, vmax=1)
        ax.set_title(label, fontsize=7, pad=2, fontweight="normal")


def draw_row_B(axes, beta_4d, conditions, model_rdms):
    """
    B1: LOC neural RDM + shape box overlay
    B2: LOC rank scatter vs shape model
    B3: SMA neural RDM + epoch box overlay
    B4: SMA rank scatter vs epoch model
    """
    ax_loc_rdm, ax_loc_scat, ax_sma_rdm, ax_sma_scat = axes

    n_cond = len(conditions)
    meta   = pd.DataFrame([parse_condition(c) for c in conditions],
                           index=conditions)

    # ── LOC ──────────────────────────────────────────────────────────────────
    cv_loc   = mni_to_vox(LOC_MNI, beta_4d.affine)
    patt_loc = extract_sphere(beta_4d, cv_loc, RADIUS_MM)
    vec_loc  = pdist(patt_loc, metric="correlation")
    rdm_loc  = squareform(vec_loc)

    im_loc = draw_rdm(ax_loc_rdm, rdm_loc, conditions=None,
                      vmin=0, vmax=2, xticks=False)

    # Shape box overlay: for each shape, find its condition indices
    shape_colors = {"circle": "#e41a1c", "rectangle": "#377eb8",
                    "star": "#4daf4a",   "triangle": "#984ea3"}
    for shape, color in shape_colors.items():
        idx = [i for i, c in enumerate(conditions)
               if parse_condition(c)["shape"] == shape]
        if idx:
            draw_box_overlay(ax_loc_rdm, idx, idx, color=color, lw=0.8)

    ax_loc_rdm.set_title(f"LOC  {LOC_MNI}", fontsize=7, pad=2)

    # Scatter vs shape model
    m_shape = rdm_to_vec(model_rdms.get("M1_shape",
                         np.zeros((n_cond, n_cond))))
    draw_scatter_rank(ax_loc_scat, vec_loc, m_shape,
                      "shape model", color="#e41a1c")

    # ── SMA ──────────────────────────────────────────────────────────────────
    cv_sma   = mni_to_vox(SMA_MNI, beta_4d.affine)
    patt_sma = extract_sphere(beta_4d, cv_sma, RADIUS_MM)
    vec_sma  = pdist(patt_sma, metric="correlation")
    rdm_sma  = squareform(vec_sma)

    im_sma = draw_rdm(ax_sma_rdm, rdm_sma, conditions=None,
                      vmin=0, vmax=2, xticks=False)

    # Epoch box overlay: rule phase (first ~16) vs test phase (~16)
    rule_idx = [i for i, c in enumerate(conditions)
                if parse_condition(c)["epoch"] == "rule"]
    test_idx = [i for i, c in enumerate(conditions)
                if parse_condition(c)["epoch"] == "test"]
    # if rule_idx:
        # draw_box_overlay(ax_sma_rdm, rule_idx, rule_idx,
                        #  color="#ff7f00", lw=0.8)
    # if test_idx:
        # draw_box_overlay(ax_sma_rdm, test_idx, test_idx,
                        #  color="#a65628", lw=0.8)

    ax_sma_rdm.set_title(f"SMA  {SMA_MNI}", fontsize=7, pad=2)

    # Scatter vs epoch model
    m_epoch = rdm_to_vec(model_rdms.get("M4_epoch",
                          np.zeros((n_cond, n_cond))))
    draw_scatter_rank(ax_sma_scat, vec_sma, m_epoch,
                      "epoch model", color="#ff7f00")

    return im_loc, im_sma


# def draw_row_C(ax_rdm, ax_ortho, beta_4d, conditions, model_rdms):
#     """
#     C1: DLPFC neural RDM + rule-type box overlay
#     C2: Ortho map of rule_type_test rho, centred on DLPFC
#     """
#     # Neural RDM
#     cv_dlpfc   = mni_to_vox(DLPFC_MNI, beta_4d.affine)
#     patt_dlpfc = extract_sphere(beta_4d, cv_dlpfc, RADIUS_MM)
#     vec_dlpfc  = pdist(patt_dlpfc, metric="correlation")
#     rdm_dlpfc  = squareform(vec_dlpfc)
#     n_cond     = len(conditions)

#     draw_rdm(ax_rdm, rdm_dlpfc, conditions=None, vmin=0, vmax=1.5,
#              xticks=False)

#     # Rule-type boxes: ABA conditions vs ABB conditions
#     aba_idx = [i for i, c in enumerate(conditions)
#                if parse_condition(c)["rule_type"] == "ABA"]
#     abb_idx = [i for i, c in enumerate(conditions)
#                if parse_condition(c)["rule_type"] == "ABB"]
#     # if aba_idx:
#     #     draw_box_overlay(ax_rdm, aba_idx, aba_idx, color="#1b7837", lw=0.8)
#     # if abb_idx:
#     #     draw_box_overlay(ax_rdm, abb_idx, abb_idx, color="#762a83", lw=0.8)

#     ax_rdm.set_title(f"DLPFC  {DLPFC_MNI}", fontsize=7, pad=2)

#     # Spearman rho inset on RDM
#     m_rule = rdm_to_vec(model_rdms.get(
#         "M5_rule_type", np.zeros((n_cond, n_cond))))
#     rho_dlpfc, p_dlpfc = spearmanr(vec_dlpfc, m_rule)
#     p_str = "p<0.001" if p_dlpfc < 0.001 else f"p={p_dlpfc:.3f}"
#     ax_rdm.text(0.97, 0.03, f"ρ={rho_dlpfc:.2f}\n{p_str}",
#                 transform=ax_rdm.transAxes, fontsize=5.5,
#                 ha="right", va="bottom",
#                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
#                           ec="0.7", lw=0.4, alpha=0.9))

#     # Ortho map
#     # Try to load spearman_full rule_type map; fall back gracefully
#     rho_path = os.path.join(
#         RESULTS_DIR,
#         f"{FOCAL_SUBJECT}_{SPEARMAN_PREFIX}_M5_rule_type.nii.gz"
#     )
#     # also try test-phase rule model
#     if not os.path.exists(rho_path):
#         rho_path = os.path.join(
#             RESULTS_DIR,
#             f"{FOCAL_SUBJECT}_spearman_test_rule_type_test.nii.gz"
#         )

#     if os.path.exists(rho_path):
#         rho_img    = nib.load(rho_path)
#         masked_img = apply_gm_mask(rho_img)
#         # positive only for ortho
#         data = masked_img.get_fdata().copy()
#         data[data <= 0] = np.nan
#         pos_img = nib.Nifti1Image(data, masked_img.affine, masked_img.header)
#         draw_ortho(ax_ortho, pos_img, DLPFC_MNI, threshold=0.1)
#     else:
#         ax_ortho.text(0.5, 0.5, "searchlight map\nnot found",
#                       ha="center", va="center", transform=ax_ortho.transAxes,
#                       fontsize=7, color="gray")
#         ax_ortho.axis("off")

def draw_row_C(ax_rdm, ax_scat, ax_ortho, ax_cbar, beta_4d, conditions, model_rdms):
    """
    C1: DLPFC neural RDM + rho inset
    C2: rank scatter (neural vs rule_type model)
    C3: ortho map thresholded at 0.1
    """
    # ── Compute neural RDM (full conditions) ─────────────────────────────────
    cv_dlpfc   = mni_to_vox(DLPFC_MNI, beta_4d.affine)
    patt_dlpfc = extract_sphere(beta_4d, cv_dlpfc, RADIUS_MM)
    vec_dlpfc  = pdist(patt_dlpfc, metric="correlation")
    rdm_dlpfc  = squareform(vec_dlpfc)
    n_cond     = len(conditions)

    # ── C1: RDM ──────────────────────────────────────────────────────────────
    im_dlpfc = draw_rdm(ax_rdm, rdm_dlpfc, conditions=None, vmin=0, vmax=1.5, xticks=False)
    ax_rdm.set_title(f"DLPFC  {DLPFC_MNI}", fontsize=7, pad=2)

    m_rule     = rdm_to_vec(model_rdms.get("M5_rule_type",
                             np.zeros((n_cond, n_cond))))
    # rho_v, p_v = spearmanr(vec_dlpfc, m_rule)

    add_cbar_in_axis(fig, ax_cbar, mappable=im_dlpfc, ticksize=6)

    # ── C2: rank scatter (neural vs rule_type) ────────────────────────────────
    # draw_scatter_rank(ax_scat, vec_dlpfc, m_rule,
                #  "rule type model rank", color="#1b7837")
    mask       = m_rule != 0.5                          # True for informative pairs
    draw_scatter_rank(ax_scat,
                      vec_dlpfc[mask], m_rule[mask],
                      "rule type model", color="#1b7837")
    rho_v, p_v = spearmanr(vec_dlpfc[mask], m_rule[mask])
    p_str      = "p<0.001" if p_v < 0.001 else f"p={p_v:.3f}"
    # ax_rdm.text(0.97, 0.03, f"ρ={rho_v:.2f}\n{p_str}",
                # transform=ax_rdm.transAxes, fontsize=5.5,
                # ha="right", va="bottom",
                # bbox=dict(boxstyle="round,pad=0.2", fc="white",
                        #   ec="0.7", lw=0.4, alpha=0.9))
    # ── C3: ortho map ─────────────────────────────────────────────────────────
    rho_path = os.path.join(RESULTS_DIR,
                             f"{FOCAL_SUBJECT}_{SPEARMAN_PREFIX}_M5_rule_type.nii.gz")
    if not os.path.exists(rho_path):
        rho_path = os.path.join(RESULTS_DIR,
                                 f"{FOCAL_SUBJECT}_spearman_test_rule_type_test.nii.gz")

    if os.path.exists(rho_path):
        masked = apply_gm_mask(nib.load(rho_path))
        d      = masked.get_fdata().copy()
        d[d <= 0] = np.nan
        draw_ortho(ax_ortho,
                   nib.Nifti1Image(d, masked.affine, masked.header),
                   DLPFC_MNI, threshold=0.1)
    else:
        ax_ortho.text(0.5, 0.5, "searchlight map\nnot found",
                      ha="center", va="center",
                      transform=ax_ortho.transAxes, fontsize=7, color="gray")
        ax_ortho.axis("off")

    return im_dlpfc
def draw_row_D(axes, model_rdms, conditions):
    regions = [
        ("LOC",   LOC_MNI,   axes[0]),
        ("SMA",   SMA_MNI,   axes[1]),
        ("DLPFC", DLPFC_MNI, axes[2]),
    ]

    # Load test-phase model RDMs for shape_test
    try:
        _, model_rdms_test = load_model_rdms("test")
    except FileNotFoundError:
        model_rdms_test = {}

    # Full-condition models
    FULL_KEYS   = ["M1_shape", "M4_epoch", "M5_rule_type", "M2_abstract_role"]
    FULL_LABELS = ["shape\n(full)", "epoch", "rule type", "abstract role"]

    # # Test-phase models appended
    # TEST_KEYS   = ["shape_test"]
    TEST_KEYS = []
    TEST_LABELS = []
    # TEST_LABELS = ["shape\n(test)"]

    ALL_KEYS   = FULL_KEYS  # + TEST_KEYS
    ALL_LABELS = FULL_LABELS #+ TEST_LABELS
    n_models   = len(ALL_KEYS)

    # Get test-phase condition indices (same for all subjects given shared design)
    test_indices = [i for i, c in enumerate(conditions)
                    if parse_condition(c)["epoch"] == "test"]

    subject_colors = {
        "sub-001": "#1a9850",
        "sub-002": "#4393c3",
        "sub-008": "#d6604d",
    }
    dot_offset = np.linspace(-0.15, 0.15, len(ALL_SUBJECTS))

    for region_name, region_mni, ax in regions:

        vals = {s: [] for s in ALL_SUBJECTS}

        for subject in ALL_SUBJECTS:
            # Load this subject's betas — use local variable, don't shadow outer conditions
            b4d, subj_conds = load_betas(subject)
            cv              = mni_to_vox(region_mni, b4d.affine)
            patt            = extract_sphere(b4d, cv, RADIUS_MM)
            # patt: (n_cond, k_voxels)

            # ── Full-condition rhos ───────────────────────────────────────────
            vec_full = pdist(patt, metric="correlation")   # (n_cond*(n_cond-1)/2,)
            n_full   = len(subj_conds)

            for key in FULL_KEYS:
                rdm_full = model_rdms.get(key)
                if rdm_full is None:
                    vals[subject].append(np.nan)
                    continue
                # ensure square, then condense
                rdm_sq  = squareform(rdm_full) if rdm_full.ndim == 1 else rdm_full
                mvec    = squareform(rdm_sq)    # condensed, same order as pdist
                if np.nanstd(mvec) < 1e-10:
                    vals[subject].append(np.nan)
                else:
                    rho, _ = spearmanr(vec_full, mvec)
                    vals[subject].append(rho)

            # ── Test-phase rhos ───────────────────────────────────────────────
            # Re-derive test indices for this subject's condition list
            t_idx    = [i for i, c in enumerate(subj_conds)
                        if parse_condition(c)["epoch"] == "test"]
            patt_t   = patt[t_idx, :]                      # (n_test, k_voxels)
            vec_test = pdist(patt_t, metric="correlation")  # (n_test*(n_test-1)/2,)

            for key in TEST_KEYS:
                rdm_test = model_rdms_test.get(key)
                if rdm_test is None:
                    vals[subject].append(np.nan)
                    continue
                # model_rdms_test already indexed to test conditions
                rdm_sq = squareform(rdm_test) if rdm_test.ndim == 1 else rdm_test
                mvec   = squareform(rdm_sq)
                if np.nanstd(mvec) < 1e-10 or len(mvec) != len(vec_test):
                    vals[subject].append(np.nan)
                else:
                    rho, _ = spearmanr(vec_test, mvec)
                    vals[subject].append(rho)

        # ── Plot ──────────────────────────────────────────────────────────────
        all_vals = np.array([vals[s] for s in ALL_SUBJECTS])  # (n_sub, n_models)
        mean_v   = np.nanmean(all_vals, axis=0)

        bar_colors = ["#d73027" if (not np.isnan(v) and v > 0) else "#4575b4"
                      for v in mean_v]
        x = np.arange(n_models)
        ax.bar(x, np.where(np.isnan(mean_v), 0, mean_v),
               color=bar_colors, edgecolor="k",
               linewidth=0.4, width=0.55, alpha=0.65, zorder=2)
        ax.axhline(0, color="k", linewidth=0.4, zorder=1)

        for s_idx, subject in enumerate(ALL_SUBJECTS):
            label = SUBJECT_LABELS.get(subject, subject)
            ax.scatter(
                x + dot_offset[s_idx], vals[subject],
                s=14, color=subject_colors.get(subject, "gray"),
                zorder=3, linewidths=0.3, edgecolors="white",
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ALL_LABELS, rotation=35, ha="right", fontsize=6)
        ax.set_yticks([])
        ax.set_ylabel("Spearman ρ", fontsize=7, labelpad=2)
        ax.set_title(region_name, fontsize=7, pad=2)
        if region_name == "DLPFC":
            ax.legend(fontsize=5.5, frameon=False,
                      loc="upper left", handletextpad=0.3)
# def draw_row_D(axes, model_rdms, conditions):
#     """
#     D1: LOC coefficients  D2: SMA coefficients  D3: DLPFC coefficients
#     Each panel: bar = mean across subjects, dots = individual subjects.
#     Coefficients = Spearman rho from searchlight NIfTI at region coordinate.
#     """
#     regions = [
#         ("LOC",   LOC_MNI,   axes[0]),
#         ("SMA",   SMA_MNI,   axes[1]),
#         ("DLPFC", DLPFC_MNI, axes[2]),
#     ]
#     model_keys = list(model_rdms.keys())
#     n_models   = len(model_keys)

#     subject_colors = {
#         "sub-001": "#1a9850",
#         "sub-002": "#4393c3",
#         "sub-008": "#d6604d",
#     }
#     dot_offset = np.linspace(-0.15, 0.15, len(ALL_SUBJECTS))

#     for region_name, region_mni, ax in regions:
#         # recalculate teh rhos at the region coordinate for all subjects, since we may have

#         # Recalculate rhos at the region coordinate for all subjects

#         # Collect rho at region voxel for each subject × model
#         vals = {s: [] for s in ALL_SUBJECTS}

#         for subject in ALL_SUBJECTS:
#             beta_4d, conditions = load_betas(subject)
#             cv = mni_to_vox(region_mni, beta_4d.affine)
#             patt = extract_sphere(beta_4d, cv, RADIUS_MM)
#             vec_neural = pdist(patt, metric="correlation")

#             for key in ['M1_shape', 'M4_epoch', 'M5_rule_type', 'M2_abstract_role']:
#                 model_vec = rdm_to_vec(model_rdms.get(key, np.zeros((len(conditions), len(conditions)))))
#                 rho, _ = spearmanr(vec_neural, model_vec)
#                 vals[subject].append(rho)
            
#             # Filter test phase conditions
#             test_indices = [i for i, c in enumerate(conditions) if parse_condition(c)["epoch"] == "test"]
#             patt_test = patt[:, test_indices]
#             vec_test = pdist(patt_test, metric="correlation")

#             for key in ['shape_test']:
#                 model_vec = rdm_to_vec(model_rdms.get(key, np.zeros((len(conditions), len(conditions)))))
#                 model_vec_test = model_vec[np.ix_(test_indices, test_indices)]
#                 rho, _ = spearmanr(vec_test, rdm_to_vec(model_vec_test))
#                 vals[subject].append(rho)

#         # Mean bar
#         all_vals = np.array([vals[s] for s in ALL_SUBJECTS])   # (n_sub, n_mod)
#         mean_v   = np.nanmean(all_vals, axis=0)
#         bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in mean_v]

#         x = np.arange(5)
#         bars = ax.bar(x, mean_v, color=bar_colors, edgecolor="k",
#                       linewidth=0.4, width=0.55, alpha=0.65, zorder=2)
#         ax.axhline(0, color="k", linewidth=0.4, zorder=1)

#         # Individual dots
#         for s_idx, subject in enumerate(ALL_SUBJECTS):
#             label = SUBJECT_LABELS.get(subject, subject)
#             ax.scatter(
#                 x + dot_offset[s_idx], vals[subject],
#                 s=14, color=subject_colors.get(subject, "gray"),
#                 zorder=3, linewidths=0.3, edgecolors="white",
#                 label=label,
#             )

#         ax.set_xticks(x)
#         ax.set_xticklabels(
#             [k.replace("M1_", "").replace("M2_", "").replace("M3_", "")
#               .replace("M4_", "").replace("M5_", "").replace("M6_", "")
#               .replace("_full", "")
#              for k in ['M1_shape', 'M4_epoch', 'M5_rule_type', 'M2_abstract_role', 'shape_test']],
#             rotation=35, ha="right", fontsize=6,
#         )
#         ax.set_yticks([])
#         ax.set_ylabel("Spearman ρ", fontsize=7, labelpad=2)
#         ax.set_title(region_name, fontsize=7, pad=2)

#         if region_name == "DLPFC":
#             ax.legend(fontsize=5.5, frameon=False,
#                       loc="upper left", handletextpad=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_figure():
    print("Loading data ...")
    beta_4d, conditions = load_betas(FOCAL_SUBJECT)
    _, model_rdms       = load_model_rdms("full")

    print(f"  {FOCAL_SUBJECT} → {len(conditions)} conditions, "
          f"beta shape {beta_4d.shape}")
    print(f"  Model RDMs: {list(model_rdms.keys())}")

    # ── Figure + GridSpec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Outer GridSpec: 4 rows
    outer = gridspec.GridSpec(
        4, 1,
        figure     = fig,
        height_ratios = [0.2, 0.241, 0.241, 0.2],
        # hspace     = 0.48,
        left       = 0.06,
        right      = 0.97,
        top        = 0.97,
        bottom     = 0.05,
    )

    # ── ROW A: 4 model RDMs ─────────────────────────────────────────────────
    gs_a = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[0],
        # wspace=0.35,
        width_ratios=[1, 1, 1, 1, 0.06],
    )
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    ax_a_cbar = fig.add_subplot(gs_a[0, 4])
    print("Drawing row A ...")
    draw_row_A(axes_a, model_rdms, conditions)

    # Row A: shared model dissimilarity colorbar (blue=0, red=1)
    add_cbar_in_axis(
        fig,
        ax_a_cbar,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        label="Model Representational Dissimilarity",
        ticksize=6,
    )

    # ── ROW B: LOC + SMA (RDM + scatter each) ───────────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 6, subplot_spec=outer[1],
        wspace=0.35,
        width_ratios=[1.1, 0.1, 0.9, 1.1, 0.1, 0.9],
    )
    ax_loc_rdm  = fig.add_subplot(gs_b[0, 0])
    ax_loc_cbar = fig.add_subplot(gs_b[0, 1])
    ax_loc_scat = fig.add_subplot(gs_b[0, 2])
    ax_sma_rdm  = fig.add_subplot(gs_b[0, 3])
    ax_sma_cbar = fig.add_subplot(gs_b[0, 4])
    ax_sma_scat = fig.add_subplot(gs_b[0, 5])

    axes_b = [ax_loc_rdm, ax_loc_scat, ax_sma_rdm, ax_sma_scat]
    print("Drawing row B ...")
    im_loc, im_sma = draw_row_B(axes_b, beta_4d, conditions, model_rdms)

    # Row B: add colorbars for the RDM heatmaps
    add_cbar_in_axis(fig, ax_loc_cbar, mappable=im_loc, ticksize=6)
    add_cbar_in_axis(fig, ax_sma_cbar, mappable=im_sma, ticksize=6)
    # ── Row C: RDM | scatter | ortho ─────────────────────────────────────────────
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2],
        wspace=0.35,
        width_ratios=[1.2, 0.9, 2.4],
    )
    ax_c_rdm   = fig.add_subplot(gs_c[0, 0])
    # ax_c_cbar  = fig.add_subplot(gs_c[0, 1])
    ax_c_cbar = ax_c_rdm.inset_axes([1.02, 0, 0.05, 1], transform=ax_c_rdm.transAxes)
    ax_c_scat  = fig.add_subplot(gs_c[0, 1])
    ax_c_ortho = fig.add_subplot(gs_c[0, 2])
    im_dlpfc = draw_row_C(ax_c_rdm, ax_c_scat, ax_c_ortho, ax_c_cbar, beta_4d, conditions, model_rdms)

    # Row C: add colorbar for the DLPFC neural RDM heatmap
    # # ── ROW C: DLPFC RDM + ortho ─────────────────────────────────────────────
    # gs_c = gridspec.GridSpecFromSubplotSpec(
    #     1, 2, subplot_spec=outer[2],
    #     wspace=0.25,
    #     width_ratios=[0.7, 2.3],
    # )
    # ax_c_rdm   = fig.add_subplot(gs_c[0, 0])
    # ax_c_ortho = fig.add_subplot(gs_c[0, 1])
    # print("Drawing row C ...")
    # draw_row_C(ax_c_rdm, ax_c_ortho, beta_4d, conditions, model_rdms)

    # ── ROW D: 3 coefficient panels ──────────────────────────────────────────
    gs_d = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[3],
        # wspace=0.45,
    )
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(3)]
    print("Drawing row D ...")
    # draw_row_D(axes_d, model_rdms, conditions)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    # also save PNG for quick preview
    fig.savefig(OUT_PATH.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved: {OUT_PATH.replace('.pdf', '.png')}")


if __name__ == "__main__":
    build_figure()

