"""
MEG Decoding Toolkit
====================

Feature extractors
------------------
RawAmplitudeFeature   — per-timepoint sensor pattern (King & Dehaene-style)
TimeFrequencyFeature  — spectrogram / wavelet power
RiemannianCovarianceFeature — trial covariances on the SPD manifold

Decoders
--------
RidgeLogisticDecoder  — L2 logistic regression (sklearn)
SVMDecoder            — linear or kernel SVM (sklearn)
DeepMLPDecoder        — PyTorch MLP, CUDA-accelerated

Cross-validation
----------------
CVSplitter            — stratified k-fold wrapper
CVResult              — dataclass holding per-fold scores
cross_validate        — main CV loop (no-leakage guarantee)

Experiment
----------
DecodingExperiment    — thin convenience wrapper
"""

from .features import (
    Feature,
    RawAmplitudeFeature,
    TimeFrequencyFeature,
    RiemannianCovarianceFeature,
)
from .decoders import (
    Decoder,
    RidgeLogisticDecoder,
    SVMDecoder,
    DeepMLPDecoder,
)
from .cv import (
    CVSplitter,
    CVResult,
    cross_validate,
)
from .pseudo_trials import make_pseudo_trials
from .conditions import (
    assign_conditions,
    condition_counts,
    PHASE_OFFSETS,
    RULE_PHASES,
    TEST_PHASES,
    STIM_CATEGORIES,
)
from .rsa import (
    RSAResult,
    build_model_rdm,
    check_model_rdm_collinearity,
    time_resolved_rsa,
)
from .encoding import (
    EncodingResult,
    build_feature_space,
    fit_encoding_model,
    nested_comparison,
)
from .cross_modal import cross_modal_rsa, cross_modal_rsa_fast
from .experiment import DecodingExperiment
from .viz import get_sensor_positions, plot_sensor_map, plot_sensor_map_3d

__all__ = [
    # features
    "Feature",
    "RawAmplitudeFeature",
    "TimeFrequencyFeature",
    "RiemannianCovarianceFeature",
    # decoders
    "Decoder",
    "RidgeLogisticDecoder",
    "SVMDecoder",
    "DeepMLPDecoder",
    # cv
    "CVSplitter",
    "CVResult",
    "cross_validate",
    # pseudo-trials
    "make_pseudo_trials",
    # conditions
    "assign_conditions",
    "condition_counts",
    "PHASE_OFFSETS",
    "RULE_PHASES",
    "TEST_PHASES",
    "STIM_CATEGORIES",
    # rsa
    "RSAResult",
    "build_model_rdm",
    "check_model_rdm_collinearity",
    "time_resolved_rsa",
    # encoding
    "EncodingResult",
    "build_feature_space",
    "fit_encoding_model",
    "nested_comparison",
    # cross-modal
    "cross_modal_rsa",
    "cross_modal_rsa_fast",
    # experiment
    "DecodingExperiment",
    "get_sensor_positions",
    "plot_sensor_map",
    "plot_sensor_map_3d",
]
