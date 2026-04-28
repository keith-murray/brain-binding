"""
High-level experiment orchestration.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .features import Feature
from .decoders import Decoder
from .cv import CVSplitter, CVResult, cross_validate


class DecodingExperiment:
    """Convenience wrapper binding a feature extractor, decoder factory, and CV splitter.

    Usage
    -----
    exp = DecodingExperiment(
        feature=RawAmplitudeFeature(...),
        decoder_factory=lambda: RidgeLogisticDecoder(C=1.0),
        cv=CVSplitter(n_splits=5),
        metric="roc_auc",
    )
    result = exp.run(X, y, sfreq=400, times=epochs.times)
    """

    def __init__(
        self,
        feature: Feature,
        decoder_factory: Callable[[], Decoder],
        cv: CVSplitter,
        metric: str = "accuracy",
    ):
        self.feature         = feature
        self.decoder_factory = decoder_factory
        self.cv              = cv
        self.metric          = metric

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sfreq: float,
        times: np.ndarray,
        return_predictions: bool = False,
        return_train_scores: bool = False,
    ) -> CVResult:
        """Run the full CV pipeline and return a CVResult."""
        return cross_validate(
            feature=self.feature,
            decoder_factory=self.decoder_factory,
            X=X,
            y=y,
            sfreq=sfreq,
            times=times,
            cv=self.cv,
            metric=self.metric,
            return_predictions=return_predictions,
            return_train_scores=return_train_scores,
        )
