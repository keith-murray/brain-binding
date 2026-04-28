"""
Pseudo-trial averaging for SNR boosting in MEG decoding.

Two construction modes:
  disjoint  — non-overlapping groups of k original trials per class
  bootstrap — k trials sampled without replacement per pseudo-trial,
              groups may share original trials across pseudo-trials
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def make_pseudo_trials(
    X: np.ndarray,                          # (n_trials, n_channels, n_times)
    y: np.ndarray,                          # (n_trials,) integer labels
    k: int = 5,                             # trials per pseudo-trial
    mode: str = "disjoint",                 # "disjoint" | "bootstrap"
    n_pseudo_per_class: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pseudo-trials by averaging within-class groups of original trials.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times)
    y : (n_trials,) integer labels
    k : trials per pseudo-trial
    mode : "disjoint" | "bootstrap"
    n_pseudo_per_class : number of pseudo-trials per class.
        disjoint  — capped at floor(N_c / k); raises ValueError if set higher.
        bootstrap — defaults to floor(N_c / k) if None.
    rng : numpy random Generator; created from seed 0 if None.

    Returns
    -------
    Xp : (n_pseudo, n_channels, n_times)
    yp : (n_pseudo,)

    Notes
    -----
    MUST be called inside the CV fold loop, not before splitting, to prevent
    leakage of original trials across train/test boundaries.
    """
    if mode not in ("disjoint", "bootstrap"):
        raise ValueError(f"mode must be 'disjoint' or 'bootstrap', got {mode!r}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if rng is None:
        rng = np.random.default_rng(0)

    classes = np.unique(y)
    Xp_parts = []
    yp_parts = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        n_c = len(idx)
        max_pseudo = n_c // k

        if max_pseudo == 0:
            raise ValueError(
                f"Class {cls} has only {n_c} trials but k={k}; "
                "not enough trials to form even one pseudo-trial."
            )

        if mode == "disjoint":
            if n_pseudo_per_class is not None:
                if n_pseudo_per_class > max_pseudo:
                    raise ValueError(
                        f"n_pseudo_per_class={n_pseudo_per_class} exceeds the "
                        f"maximum {max_pseudo} disjoint groups for class {cls} "
                        f"(N_c={n_c}, k={k}). Use bootstrap mode for more pseudo-trials."
                    )
                m = n_pseudo_per_class
            else:
                m = max_pseudo

            shuffled = rng.permutation(idx)
            # Take exactly m*k indices, reshape into (m, k) groups
            groups = shuffled[: m * k].reshape(m, k)

        else:  # bootstrap
            m = n_pseudo_per_class if n_pseudo_per_class is not None else max_pseudo
            # Each group: k distinct trials drawn without replacement from class
            groups = np.stack(
                [rng.choice(idx, size=k, replace=False) for _ in range(m)],
                axis=0,
            )  # (m, k)

        # Average the k original trials in each group → pseudo-trial
        pseudo = X[groups].mean(axis=1)  # (m, n_channels, n_times)
        Xp_parts.append(pseudo)
        yp_parts.append(np.full(m, cls, dtype=y.dtype))

    Xp = np.concatenate(Xp_parts, axis=0)
    yp = np.concatenate(yp_parts, axis=0)
    return Xp, yp
