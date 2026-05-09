"""
Decoder classes for MEG decoding toolkit.

Each Decoder trains one classifier per "model slot" (e.g., one per timepoint).
The fit/predict/score API mirrors sklearn but operates over the n_models axis.

Parallelization:
  RidgeLogisticDecoder and SVMDecoder use joblib.Parallel over model slots.
  DeepMLPDecoder runs PyTorch models sequentially on CUDA (faster than
  spawning 200+ subprocesses for small GPU jobs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Decoder(ABC):
    """Fits one classifier per model slot."""

    n_models: int
    models: List
    is_fitted: bool = False

    def __init__(self, **config):
        pass

    @abstractmethod
    def fit(self, features: np.ndarray, y: np.ndarray) -> "Decoder":
        """features: (n_models, n_epochs, feature_dim), y: (n_epochs,)"""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns (n_models, n_epochs)."""
        ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Returns (n_models, n_epochs, n_classes). Raise if unsupported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba.")

    def score(
        self,
        features: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
    ) -> np.ndarray:
        """Returns (n_models,). metric: 'accuracy' | 'roc_auc' | 'balanced_accuracy'"""
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, balanced_accuracy_score
        )

        if metric == "accuracy":
            preds = self.predict(features)
            return np.array([accuracy_score(y, preds[m]) for m in range(self.n_models)])
        elif metric == "balanced_accuracy":
            preds = self.predict(features)
            return np.array([balanced_accuracy_score(y, preds[m]) for m in range(self.n_models)])
        elif metric == "roc_auc":
            try:
                probas = self.predict_proba(features)  # (n_models, n_epochs, n_classes)
                n_classes = probas.shape[-1]
                if n_classes == 2:
                    return np.array([
                        roc_auc_score(y, probas[m, :, 1]) for m in range(self.n_models)
                    ])
                else:
                    return np.array([
                        roc_auc_score(y, probas[m], multi_class="ovr") for m in range(self.n_models)
                    ])
            except NotImplementedError:
                # Fall back to accuracy if proba is not supported
                import warnings
                warnings.warn(
                    f"{self.__class__.__name__} does not support predict_proba; "
                    "falling back to accuracy for roc_auc."
                )
                preds = self.predict(features)
                return np.array([accuracy_score(y, preds[m]) for m in range(self.n_models)])
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_one_sklearn(clf_factory, X_m, y):
    """Fit a single sklearn classifier; used as a joblib task."""
    clf = clf_factory()
    clf.fit(X_m, y)
    return clf


# ---------------------------------------------------------------------------
# Ridge logistic regression
# ---------------------------------------------------------------------------

class RidgeLogisticDecoder(Decoder):
    """L2-regularized logistic regression via sklearn, parallelized with joblib."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        solver: str = "lbfgs",
        n_jobs: int = -1,
    ):
        self.C            = C
        self.class_weight = class_weight
        self.max_iter     = max_iter
        self.solver       = solver
        self.n_jobs       = n_jobs
        self.models: List = []

    def _factory(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            # penalty="l2",
            l1_ratio=0,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver=self.solver,
        )

    def fit(self, features: np.ndarray, y: np.ndarray) -> "RidgeLogisticDecoder":
        from joblib import Parallel, delayed

        self.n_models = features.shape[0]
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_sklearn)(self._factory, features[m], y)
            for m in range(self.n_models)
        )
        self.is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.models[m].predict(features[m]) for m in range(self.n_models)])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.models[m].predict_proba(features[m]) for m in range(self.n_models)],
            axis=0,
        )


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

class SVMDecoder(Decoder):
    """Linear or kernel SVM. LinearSVC is used for kernel='linear' (faster)."""

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = -1,
    ):
        self.kernel       = kernel
        self.C            = C
        self.class_weight = class_weight
        self.n_jobs       = n_jobs
        self.models: List = []
        self._use_linear  = kernel == "linear"

    def _factory(self):
        if self._use_linear:
            from sklearn.svm import LinearSVC
            return LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=2000)
        else:
            from sklearn.svm import SVC
            return SVC(
                kernel=self.kernel, C=self.C,
                class_weight=self.class_weight, probability=True,
            )

    def fit(self, features: np.ndarray, y: np.ndarray) -> "SVMDecoder":
        from joblib import Parallel, delayed

        self.n_models = features.shape[0]
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_sklearn)(self._factory, features[m], y)
            for m in range(self.n_models)
        )
        self.is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.models[m].predict(features[m]) for m in range(self.n_models)])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._use_linear:
            raise NotImplementedError(
                "LinearSVC does not support predict_proba. "
                "Use kernel != 'linear' or switch to RidgeLogisticDecoder."
            )
        return np.stack(
            [self.models[m].predict_proba(features[m]) for m in range(self.n_models)],
            axis=0,
        )


# ---------------------------------------------------------------------------
# Deep MLP (PyTorch)
# ---------------------------------------------------------------------------

class DeepMLPDecoder(Decoder):
    """Small MLP per model slot, trained with Adam + L2 weight decay on CUDA/CPU.

    One network is trained per model slot (e.g., one per timepoint). For large
    n_models this can be slow; a shared-backbone extension is left for v2.
    """

    def __init__(
        self,
        hidden_dims: List[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.3,
        weight_decay: float = 1e-3,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        val_fraction: float = 0.15,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        import torch

        self.hidden_dims             = list(hidden_dims)
        self.activation              = activation
        self.dropout                 = dropout
        self.weight_decay            = weight_decay
        self.learning_rate           = learning_rate
        self.batch_size              = batch_size
        self.max_epochs              = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction            = val_fraction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.models: List = []

    # ------------------------------------------------------------------
    def _build_net(self, in_dim: int, n_classes: int):
        import torch.nn as nn

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        Act = act_map.get(self.activation, nn.ReLU)

        layers = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), Act(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def _train_one(self, X_m: np.ndarray, y: np.ndarray):
        """Train a single MLP on (n_epochs, feature_dim) data. Returns trained net."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device)
        n_samples, feat_dim = X_m.shape
        n_classes = len(np.unique(y))

        # Train/val split (stratified-ish via random shuffle)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n_samples)
        n_val = max(1, int(n_samples * self.val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.tensor(X_m[train_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(y[train_idx],  dtype=torch.long,    device=device)
        X_va = torch.tensor(X_m[val_idx],  dtype=torch.float32, device=device)
        y_va = torch.tensor(y[val_idx],    dtype=torch.long,    device=device)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        net = self._build_net(feat_dim, n_classes).to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation loss
            net.eval()
            with torch.no_grad():
                val_loss = criterion(net(X_va), y_va).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, y: np.ndarray) -> "DeepMLPDecoder":
        self.n_models = features.shape[0]
        self.models = [
            self._train_one(features[m], y) for m in range(self.n_models)
        ]
        self.is_fitted = True
        return self

    def _infer(self, features: np.ndarray) -> np.ndarray:
        """Run inference for all model slots; returns (n_models, n_epochs, n_classes) logits."""
        import torch

        device = torch.device(self.device)
        all_logits = []
        for m in range(self.n_models):
            X_t = torch.tensor(features[m], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = self.models[m](X_t).cpu().numpy()  # (n_epochs, n_classes)
            all_logits.append(logits)
        return np.stack(all_logits, axis=0)  # (n_models, n_epochs, n_classes)

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = self._infer(features)
        return logits.argmax(axis=-1)  # (n_models, n_epochs)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        import torch

        logits = self._infer(features)
        # Softmax along class axis
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
