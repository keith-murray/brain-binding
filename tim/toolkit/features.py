"""
Feature extraction classes for MEG decoding.

Every Feature produces output of shape (n_models, n_epochs, feature_dim).
The fit/transform split is mandatory: fit() runs only on training data so
that any learned statistics (z-score params, Riemannian mean) avoid leakage.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Feature(ABC):
    """Abstract base for all feature extractors.

    Subclasses must implement fit() and transform(). Stateless features
    (raw amplitude, TF power) still use fit() to compute z-score statistics
    on training data only.
    """

    feature_dim: int
    n_models: int
    model_coords: dict

    def __init__(self, **config):
        pass

    def fit(self, X: np.ndarray, sfreq: float, times: np.ndarray) -> "Feature":
        """Compute any statistics needed for transform; must use training data only."""
        self._sfreq = sfreq
        self._times = times
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return (n_models, n_epochs, feature_dim) float32 array."""
        ...

    def fit_transform(self, X: np.ndarray, sfreq: float, times: np.ndarray) -> np.ndarray:
        return self.fit(X, sfreq, times).transform(X)

    def describe(self) -> str:
        return self.__class__.__name__

    def clone(self) -> "Feature":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Raw amplitude
# ---------------------------------------------------------------------------

class RawAmplitudeFeature(Feature):
    """Per-timepoint sensor pattern decoding (King & Dehaene-style MVPA).

    n_models = n_times_sub  (one decoder per timepoint)
    feature_dim = n_channels
    """

    def __init__(
        self,
        time_window: Optional[Tuple[float, float]] = None,
        decimate: int = 1,
        standardize: bool = True,
    ):
        """
        Parameters
        ----------
        time_window : (tmin, tmax) in seconds, or None for full epoch.
        decimate    : integer downsampling factor along the time axis.
        standardize : z-score per (channel, timepoint) across training epochs.
        """
        self.time_window = time_window
        self.decimate = decimate
        self.standardize = standardize

        self._mean: Optional[np.ndarray] = None   # (n_channels, n_times_sub)
        self._std:  Optional[np.ndarray] = None
        self._time_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, sfreq: float, times: np.ndarray) -> "RawAmplitudeFeature":
        super().fit(X, sfreq, times)

        # Time-window mask
        if self.time_window is not None:
            tmin, tmax = self.time_window
            self._time_mask = (times >= tmin) & (times <= tmax)
        else:
            self._time_mask = np.ones(len(times), dtype=bool)

        times_sub = times[self._time_mask][:: self.decimate]
        X_sub = X[:, :, self._time_mask][:, :, :: self.decimate]  # (n_ep, C, T_sub)

        self.n_models = len(times_sub)
        self.feature_dim = X.shape[1]
        self.model_coords = {"time_s": times_sub}

        if self.standardize:
            # Mean/std per (channel, timepoint) across training epochs
            self._mean = X_sub.mean(axis=0)           # (C, T_sub)
            self._std  = X_sub.std(axis=0)
            # self._std  = np.where(self._std < 1e-10, 1.0, self._std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_sub = X[:, :, self._time_mask][:, :, :: self.decimate]

        if self.standardize and self._mean is not None:
            X_sub = (X_sub - self._mean[np.newaxis]) / self._std[np.newaxis]

        # (n_epochs, C, T_sub) → (T_sub, n_epochs, C)
        features = np.ascontiguousarray(X_sub.transpose(2, 0, 1), dtype=np.float32)
        return features  # (n_models, n_epochs, feature_dim)

    def describe(self) -> str:
        return (
            f"RawAmplitude(window={self.time_window}, "
            f"decimate={self.decimate}, standardize={self.standardize})"
        )


# ---------------------------------------------------------------------------
# Time-frequency power
# ---------------------------------------------------------------------------

class TimeFrequencyFeature(Feature):
    """Spectrogram / wavelet-based TF power features.

    mode="stacked"  → n_models = n_times,  feature_dim = n_channels * n_freqs
    mode="separate" → n_models = n_times * n_freqs, feature_dim = n_channels
    """

    VALID_METHODS = ("morlet", "stft", "multitaper")
    VALID_OUTPUTS = ("power", "log_power", "zscore_power")
    VALID_MODES   = ("stacked", "separate")

    def __init__(
        self,
        freqs: np.ndarray,
        method: str = "morlet",
        n_cycles: Union[float, np.ndarray] = 7.0,
        mode: str = "stacked",
        output: str = "log_power",
        time_window: Optional[Tuple[float, float]] = None,
        decimate: int = 1,
        standardize: bool = True,
    ):
        assert method in self.VALID_METHODS, f"method must be one of {self.VALID_METHODS}"
        assert output in self.VALID_OUTPUTS, f"output must be one of {self.VALID_OUTPUTS}"
        assert mode   in self.VALID_MODES,   f"mode must be one of {self.VALID_MODES}"

        self.freqs       = np.asarray(freqs, dtype=np.float64)
        self.method      = method
        self.n_cycles    = n_cycles
        self.mode        = mode
        self.output      = output
        self.time_window = time_window
        self.decimate    = decimate
        self.standardize = standardize

        self._mean: Optional[np.ndarray] = None   # (C, F, T_sub) fit-time stats
        self._std:  Optional[np.ndarray] = None
        self._time_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _compute_tfr(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        """Return TFR power (n_epochs, n_channels, n_freqs, n_times), float64."""
        import mne.time_frequency as mtf

        if self.method == "morlet":
            power = mtf.tfr_array_morlet(
                X, sfreq=sfreq, freqs=self.freqs,
                n_cycles=self.n_cycles, output="power", verbose=False,
            )
        elif self.method == "multitaper":
            power = mtf.tfr_array_multitaper(
                X, sfreq=sfreq, freqs=self.freqs,
                n_cycles=self.n_cycles, output="power", verbose=False,
            )
        elif self.method == "stft":
            # MNE doesn't expose a raw STFT array function; fall back to morlet.
            import warnings
            warnings.warn("stft method not directly supported; falling back to morlet.")
            power = mtf.tfr_array_morlet(
                X, sfreq=sfreq, freqs=self.freqs,
                n_cycles=self.n_cycles, output="power", verbose=False,
            )
        return power  # (n_epochs, C, F, T)

    def _apply_output(self, power: np.ndarray) -> np.ndarray:
        """Apply output transform in place (float32)."""
        power = power.astype(np.float32)
        if self.output == "log_power":
            # np.log1p(power, out=power)
            np.log(power, out=power)
        elif self.output == "zscore_power":
            # z-score over time within each (epoch, channel, freq)
            m = power.mean(axis=-1, keepdims=True)
            s = power.std(axis=-1, keepdims=True)
            s = np.where(s < 1e-10, 1.0, s)
            power = (power - m) / s
        # "power" → return as is
        return power

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, sfreq: float, times: np.ndarray) -> "TimeFrequencyFeature":
        super().fit(X, sfreq, times)

        # Time-window mask & decimation
        if self.time_window is not None:
            tmin, tmax = self.time_window
            self._time_mask = (times >= tmin) & (times <= tmax)
        else:
            self._time_mask = np.ones(len(times), dtype=bool)

        times_sub = times[self._time_mask][:: self.decimate]
        n_ch = X.shape[1]
        n_f  = len(self.freqs)
        n_t  = len(times_sub)

        if self.mode == "stacked":
            self.n_models   = n_t
            self.feature_dim = n_ch * n_f
            self.model_coords = {"time_s": times_sub, "freq_hz": self.freqs}
        else:  # "separate"
            self.n_models   = n_t * n_f
            self.feature_dim = n_ch
            time_grid, freq_grid = np.meshgrid(times_sub, self.freqs, indexing="ij")
            self.model_coords = {
                "time_s":  time_grid.ravel(),
                "freq_hz": freq_grid.ravel(),
            }

        if self.standardize:
            power = self._compute_tfr(X, sfreq)             # (E, C, F, T)
            power = self._apply_output(power)
            power_sub = power[:, :, :, self._time_mask][:, :, :, :: self.decimate]
            # z-score stats per (C, F, T_sub) across training epochs
            self._mean = power_sub.mean(axis=0)              # (C, F, T_sub)
            self._std  = power_sub.std(axis=0)
            # self._std  = np.where(self._std < 1e-10, 1.0, self._std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        power = self._compute_tfr(X, self._sfreq)            # (E, C, F, T)
        power = self._apply_output(power)
        power_sub = power[:, :, :, self._time_mask][:, :, :, :: self.decimate]
        # power_sub: (n_epochs, C, F, T_sub)

        if self.standardize and self._mean is not None:
            power_sub = (power_sub - self._mean[np.newaxis]) / self._std[np.newaxis]

        n_ep, n_ch, n_f, n_t = power_sub.shape

        if self.mode == "stacked":
            # (n_epochs, C, F, T_sub) → (T_sub, n_epochs, C*F)
            # First merge C and F: (n_epochs, C*F, T_sub)
            merged = power_sub.reshape(n_ep, n_ch * n_f, n_t)
            features = np.ascontiguousarray(merged.transpose(2, 0, 1), dtype=np.float32)
        else:
            # (n_epochs, C, F, T_sub) → (T_sub*F, n_epochs, C)
            # Transpose to (T_sub, F, n_epochs, C) then reshape
            perm = power_sub.transpose(3, 2, 0, 1)           # (T_sub, F, n_epochs, C)
            features = np.ascontiguousarray(
                perm.reshape(n_t * n_f, n_ep, n_ch), dtype=np.float32
            )

        return features  # (n_models, n_epochs, feature_dim)

    def describe(self) -> str:
        return (
            f"TimeFrequency(method={self.method}, freqs={self.freqs[0]:.1f}-"
            f"{self.freqs[-1]:.1f}Hz, mode={self.mode}, output={self.output})"
        )


# ---------------------------------------------------------------------------
# Riemannian covariance
# ---------------------------------------------------------------------------

class RiemannianCovarianceFeature(Feature):
    """Trial-wise covariance matrices projected onto the tangent space.

    Uses pyriemann for covariance estimation and tangent-space projection.

    n_models = n_windows (1 if window_size is None)
    feature_dim = C*(C+1)/2  (upper triangle of log-mapped matrix)

    Leakage note: TangentSpace reference mean is fit on training covariances
    only. cross_validate() clones and re-fits this feature each fold.
    """

    VALID_ESTIMATORS = ("scm", "oas", "lwf")
    VALID_REFERENCES = ("riemann", "logeuclid", "identity")

    def __init__(
        self,
        window_size: Optional[float] = None,
        stride: Optional[float] = None,
        estimator: str = "oas",
        tangent_space: bool = True,
        reference: str = "riemann",
        standardize: bool = False,
        reg: float = 1e-6,
    ):
        assert estimator in self.VALID_ESTIMATORS
        assert reference in self.VALID_REFERENCES

        self.window_size  = window_size
        self.stride       = stride
        self.estimator    = estimator
        self.tangent_space = tangent_space
        self.reference    = reference
        self.standardize  = standardize
        self.reg          = reg
        self.spd_output   = not tangent_space  # hook for future SPD-aware decoders

        self._ts_list = []    # one fitted TangentSpace per window
        self._windows = []    # list of (start_idx, end_idx) pairs

        # z-score stats if standardize=True (per tangent-space dim per window)
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_windows(self, n_times: int, sfreq: float) -> list:
        """Return list of (start, stop) index pairs."""
        if self.window_size is None:
            return [(0, n_times)]
        win_s  = int(self.window_size * sfreq)
        step_s = int((self.stride or self.window_size) * sfreq)
        step_s = max(1, step_s)
        wins = []
        start = 0
        while start + win_s <= n_times:
            wins.append((start, start + win_s))
            start += step_s
        return wins if wins else [(0, n_times)]

    def _window_centers(self, windows: list, sfreq: float) -> np.ndarray:
        centers = np.array([(s + e) / 2.0 / sfreq for s, e in windows])
        return centers

    def _covariances(self, X: np.ndarray, start: int, stop: int) -> np.ndarray:
        """Return (n_epochs, C, C) covariance matrices for a window slice."""
        from pyriemann.estimation import Covariances
        X_win = X[:, :, start:stop]
        cov_est = Covariances(estimator=self.estimator)
        covs = cov_est.fit_transform(X_win)  # (n_epochs, C, C)
        if self.reg > 0:
            covs += self.reg * np.eye(covs.shape[-1])
        return covs

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, sfreq: float, times: np.ndarray) -> "RiemannianCovarianceFeature":
        super().fit(X, sfreq, times)
        from pyriemann.tangentspace import TangentSpace

        n_ep, n_ch, n_times = X.shape
        self._windows = self._build_windows(n_times, sfreq)
        centers = self._window_centers(self._windows, sfreq)

        self.n_models    = len(self._windows)
        self.feature_dim = n_ch * (n_ch + 1) // 2
        self.model_coords = {"window_center_s": centers}

        self._ts_list = []
        if self.tangent_space:
            for start, stop in self._windows:
                covs = self._covariances(X, start, stop)
                ts = TangentSpace(metric=self.reference)
                ts.fit(covs)
                self._ts_list.append(ts)

            if self.standardize:
                # Compute tangent-space vectors on training data and get stats
                vecs = []
                for idx, (start, stop) in enumerate(self._windows):
                    covs = self._covariances(X, start, stop)
                    v = self._ts_list[idx].transform(covs)  # (n_epochs, feature_dim)
                    vecs.append(v)
                vecs = np.stack(vecs, axis=0)  # (n_windows, n_epochs, feature_dim)
                self._mean = vecs.mean(axis=1)  # (n_windows, feature_dim)
                self._std  = vecs.std(axis=1)
                # self._std  = np.where(self._std < 1e-10, 1.0, self._std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.tangent_space:
            # Return (n_windows, n_epochs, C, C) — breaks 3D contract intentionally
            result = np.stack(
                [self._covariances(X, s, e) for s, e in self._windows], axis=0
            ).astype(np.float32)
            return result

        vecs = []
        for idx, (start, stop) in enumerate(self._windows):
            covs = self._covariances(X, start, stop)
            v = self._ts_list[idx].transform(covs)  # (n_epochs, feature_dim)
            vecs.append(v)

        features = np.stack(vecs, axis=0).astype(np.float32)  # (n_windows, n_epochs, fd)

        if self.standardize and self._mean is not None:
            # features = (features - self._mean[:, np.newaxis, :]) / self._std[:, np.newaxis, :]
            features = (features - self._mean[:, np.newaxis]) / 1e-16

        return features

    def describe(self) -> str:
        return (
            f"RiemannianCovariance(estimator={self.estimator}, "
            f"tangent={self.tangent_space}, reference={self.reference}, "
            f"window={self.window_size}s)"
        )
