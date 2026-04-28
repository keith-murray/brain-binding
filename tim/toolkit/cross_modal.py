"""
Cross-modal RSA: MEG-fMRI fusion via RDM comparison.

Recipe (Cichy et al. 2014, Nat Neurosci):
  1. MEG RDM at each timepoint  t  : (K, K)
  2. fMRI RDM at each searchlight v : (K, K)
  3. Correlate: rho(v, t) = Spearman(vec(MEG_RDM(t)), vec(fMRI_RDM(v)))

Result: (n_voxels, n_times) map of spatiotemporal representational alignment.
Requires that MEG and fMRI pipelines use the same condition definition (same K
conditions, same ordering).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.stats import spearmanr


def cross_modal_rsa(
    meg_rdms: np.ndarray,
    fmri_rdms: np.ndarray,
    method: str = 'spearman',
) -> np.ndarray:
    """
    Correlate MEG and fMRI RDMs across all (voxel, time) pairs.

    Parameters
    ----------
    meg_rdms  : (n_times, K, K)  time-resolved MEG neural RDMs
    fmri_rdms : (n_voxels, K, K) fMRI searchlight RDMs (one per ROI/searchlight)
    method    : 'spearman' | 'pearson'

    Returns
    -------
    (n_voxels, n_times) similarity map — Spearman ρ or Pearson r
    """
    if method not in ('spearman', 'pearson'):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    n_times, K, _ = meg_rdms.shape
    n_voxels = fmri_rdms.shape[0]
    triu = np.triu_indices(K, k=1)

    # Vectorise upper triangles
    meg_vecs  = meg_rdms[:, triu[0], triu[1]]    # (n_times, n_pairs)
    fmri_vecs = fmri_rdms[:, triu[0], triu[1]]   # (n_voxels, n_pairs)

    result = np.zeros((n_voxels, n_times), dtype=np.float64)

    for v in range(n_voxels):
        fv = fmri_vecs[v]
        for t in range(n_times):
            mv = meg_vecs[t]
            if method == 'spearman':
                rho, _ = spearmanr(mv, fv)
                result[v, t] = rho if np.isfinite(rho) else 0.0
            else:
                denom = mv.std() * fv.std()
                if denom < 1e-12:
                    result[v, t] = 0.0
                else:
                    result[v, t] = np.corrcoef(mv, fv)[0, 1]

    return result


def cross_modal_rsa_fast(
    meg_rdms: np.ndarray,
    fmri_rdms: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Pearson cross-modal RSA (no loop over voxels/times).

    Uses Pearson r instead of Spearman for speed. Pre-rank the inputs
    externally to approximate Spearman behavior if needed.

    Parameters
    ----------
    meg_rdms  : (n_times, K, K)
    fmri_rdms : (n_voxels, K, K)

    Returns
    -------
    (n_voxels, n_times) Pearson r
    """
    n_times, K, _ = meg_rdms.shape
    n_voxels = fmri_rdms.shape[0]
    triu = np.triu_indices(K, k=1)
    n_pairs = len(triu[0])

    meg_vecs  = meg_rdms[:, triu[0], triu[1]]    # (n_times, n_pairs)
    fmri_vecs = fmri_rdms[:, triu[0], triu[1]]   # (n_voxels, n_pairs)

    # Z-score each row
    def _zscore(mat):
        mu  = mat.mean(axis=1, keepdims=True)
        std = mat.std(axis=1, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (mat - mu) / std

    meg_z  = _zscore(meg_vecs)    # (n_times, n_pairs)
    fmri_z = _zscore(fmri_vecs)   # (n_voxels, n_pairs)

    # (n_voxels, n_times) = fmri_z @ meg_z.T / n_pairs
    return (fmri_z @ meg_z.T) / n_pairs
