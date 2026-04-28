"""
Cross-validation machinery for the MEG decoding toolkit.

Key guarantee: the Feature is re-fit on each training fold so that
learned statistics (z-score params, Riemannian reference mean) never
see test data. Decoder instances are also freshly created each fold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .features import Feature
from .decoders import Decoder
from .pseudo_trials import make_pseudo_trials


# ---------------------------------------------------------------------------
# CVSplitter
# ---------------------------------------------------------------------------

class CVSplitter:
    """Thin wrapper around sklearn cross-validation iterators.

    Parameters
    ----------
    n_splits     : number of CV folds
    stratified   : use StratifiedKFold to preserve class ratios
    shuffle      : shuffle data before splitting
    random_state : reproducibility seed
    group_key    : reserved for GroupKFold in future versions
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int = 42,
        group_key: Optional[str] = None,
    ):
        self.n_splits     = n_splits
        self.stratified   = stratified
        self.shuffle      = shuffle
        self.random_state = random_state
        self.group_key    = group_key  # not used in v1

    def _build_splitter(self):
        if self.stratified:
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            from sklearn.model_selection import KFold
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        splitter = self._build_splitter()
        return splitter.split(X, y)


# ---------------------------------------------------------------------------
# CVResult
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Results from a cross-validated decoding run.

    scores : (n_folds, n_models) — per-fold per-model scores
    mean_scores : (n_models,) property — average across folds
    sem_scores  : (n_models,) property — SEM across folds
    """

    scores: np.ndarray                          # (n_folds, n_models)
    train_scores: Optional[np.ndarray] = None   # (n_folds, n_models)
    predictions: Optional[List[np.ndarray]] = None  # per fold
    feature_metadata: dict = field(default_factory=dict)
    decoder_name: str = ""
    feature_name: str = ""

    @property
    def mean_scores(self) -> np.ndarray:
        """(n_models,) — mean CV score at each model slot."""
        return self.scores.mean(axis=0)

    @property
    def sem_scores(self) -> np.ndarray:
        """(n_models,) — standard error across folds."""
        return self.scores.std(axis=0) / np.sqrt(self.scores.shape[0])

    def __repr__(self) -> str:
        n_folds, n_models = self.scores.shape
        grand_mean = self.mean_scores.mean()
        return (
            f"CVResult(decoder={self.decoder_name!r}, feature={self.feature_name!r}, "
            f"folds={n_folds}, n_models={n_models}, "
            f"grand_mean={grand_mean:.4f})"
        )


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------

def cross_validate(
    feature: Feature,
    decoder_factory: Callable[[], Decoder],
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cv: CVSplitter,
    metric: str = "accuracy",
    return_predictions: bool = False,
    return_train_scores: bool = False,
    # Pseudo-trial averaging parameters
    pseudo_k: int = 1,
    pseudo_mode: str = "disjoint",
    pseudo_n_per_class: Optional[int] = None,
    pseudo_test: bool = False,
    pseudo_repetitions: int = 1,
    pseudo_seed: int = 42,
) -> CVResult:
    """Run feature-fit → decoder-fit → evaluate loop with strict no-leakage guarantee.

    Parameters
    ----------
    feature         : Feature instance (will be deep-copied and re-fit each fold)
    decoder_factory : zero-argument callable returning a fresh Decoder
    X               : (n_epochs, n_channels, n_times) float32 MEG data
    y               : (n_epochs,) integer class labels
    sfreq           : sampling rate in Hz
    times           : (n_times,) time vector in seconds
    cv              : CVSplitter instance
    metric          : 'accuracy' | 'roc_auc' | 'balanced_accuracy'
    return_predictions : store per-fold test predictions in CVResult
    return_train_scores: also score on training set each fold
    pseudo_k        : trials per pseudo-trial; 1 disables averaging (default)
    pseudo_mode     : 'disjoint' | 'bootstrap'
    pseudo_n_per_class : pseudo-trials per class per fold; None → auto
    pseudo_test     : if True, also average test-side trials (default False)
    pseudo_repetitions : random groupings to average per fold (default 1)
    pseudo_seed     : base seed; fold/rep seeds derived deterministically from it

    Returns
    -------
    CVResult with shape (n_folds, n_models) scores.
    """
    if pseudo_k == 1:
        return _cross_validate_simple(
            feature, decoder_factory, X, y, sfreq, times, cv, metric,
            return_predictions, return_train_scores,
        )
    return _cross_validate_pseudo(
        feature, decoder_factory, X, y, sfreq, times, cv, metric,
        return_predictions, return_train_scores,
        pseudo_k, pseudo_mode, pseudo_n_per_class,
        pseudo_test, pseudo_repetitions, pseudo_seed,
    )


def _cross_validate_simple(
    feature: Feature,
    decoder_factory: Callable[[], Decoder],
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cv: CVSplitter,
    metric: str,
    return_predictions: bool,
    return_train_scores: bool,
) -> CVResult:
    """Original single-trial CV loop (pseudo_k == 1 path)."""
    all_test_scores  = []
    all_train_scores = []
    all_predictions  = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        feature_fold = feature.clone()
        feature_fold.fit(X[train_idx], sfreq, times)
        f_train = feature_fold.transform(X[train_idx])
        f_test  = feature_fold.transform(X[test_idx])

        decoder = decoder_factory()
        decoder.fit(f_train, y[train_idx])

        test_scores = decoder.score(f_test, y[test_idx], metric=metric)
        all_test_scores.append(test_scores)

        if return_train_scores:
            all_train_scores.append(decoder.score(f_train, y[train_idx], metric=metric))

        if return_predictions:
            all_predictions.append(decoder.predict(f_test))

    return CVResult(
        scores=np.stack(all_test_scores, axis=0),
        train_scores=np.stack(all_train_scores, axis=0) if return_train_scores else None,
        predictions=all_predictions if return_predictions else None,
        feature_metadata=feature.model_coords if hasattr(feature, "model_coords") else {},
        decoder_name=decoder_factory().__class__.__name__,
        feature_name=feature.__class__.__name__,
    )


def _cross_validate_pseudo(
    feature: Feature,
    decoder_factory: Callable[[], Decoder],
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cv: CVSplitter,
    metric: str,
    return_predictions: bool,
    return_train_scores: bool,
    pseudo_k: int,
    pseudo_mode: str,
    pseudo_n_per_class: Optional[int],
    pseudo_test: bool,
    pseudo_repetitions: int,
    pseudo_seed: int,
) -> CVResult:
    """Pseudo-trial CV loop: construct pseudo-trials inside each fold."""
    all_test_scores  = []
    all_train_scores = []
    all_predictions  = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        rep_test_scores  = []
        rep_train_scores = []
        last_preds       = None

        for rep in range(pseudo_repetitions):
            rng = np.random.default_rng(seed=pseudo_seed + 1000 * fold_idx + rep)

            X_tr_p, y_tr_p = make_pseudo_trials(
                X[train_idx], y[train_idx],
                k=pseudo_k, mode=pseudo_mode,
                n_pseudo_per_class=pseudo_n_per_class, rng=rng,
            )

            if pseudo_test:
                X_te_p, y_te_p = make_pseudo_trials(
                    X[test_idx], y[test_idx],
                    k=pseudo_k, mode=pseudo_mode,
                    n_pseudo_per_class=pseudo_n_per_class, rng=rng,
                )
            else:
                X_te_p, y_te_p = X[test_idx], y[test_idx]

            feature_fold = feature.clone()
            feature_fold.fit(X_tr_p, sfreq, times)
            f_train = feature_fold.transform(X_tr_p)
            f_test  = feature_fold.transform(X_te_p)

            decoder = decoder_factory()
            decoder.fit(f_train, y_tr_p)

            rep_test_scores.append(decoder.score(f_test, y_te_p, metric=metric))

            if return_train_scores:
                rep_train_scores.append(decoder.score(f_train, y_tr_p, metric=metric))

            if return_predictions:
                last_preds = decoder.predict(f_test)

        fold_test_score = np.mean(np.stack(rep_test_scores, axis=0), axis=0)
        all_test_scores.append(fold_test_score)

        if return_train_scores:
            fold_train_score = np.mean(np.stack(rep_train_scores, axis=0), axis=0)
            all_train_scores.append(fold_train_score)

        if return_predictions:
            all_predictions.append(last_preds)

    return CVResult(
        scores=np.stack(all_test_scores, axis=0),
        train_scores=np.stack(all_train_scores, axis=0) if return_train_scores else None,
        predictions=all_predictions if return_predictions else None,
        feature_metadata=feature.model_coords if hasattr(feature, "model_coords") else {},
        decoder_name=decoder_factory().__class__.__name__,
        feature_name=feature.__class__.__name__,
    )
