"""PCA-based exploration of stimulus/rule encoding in epoched MEG data.

This launches a small Dash app showing a 3D scatter of trial-wise activity
projected into a PCA space. PCA bases are fit separately for each within-trial
"epoch" (fixation, rule1/2/3, transition, test1/2/3, response). The time slider
scrubs within-epoch time, while all points are projected into the selected
epoch's PCA basis.

Assumptions (based on this dataset):
- The epochs file contains 1080 epochs arranged as 120 trials × 9 epochs.
- Event code pattern per trial is:
  fixation, rule, rule, rule, transition, test, test, (test_match|test_non_match), response

If your dataset differs, adjust `EXPECTED_EVENT_CODES_BY_POS` and
`EPOCH_POS_TO_NAME`.

Run:
  conda activate test
  python tim/decoding/pca_3d_epoch_explorer.py \
    --epochs tim/ported_results/sub-001_ses-01_task-binding_epo.fif \
    --behavior data/2026-04-10/sub-001_events.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPOCH_POS_TO_NAME: Dict[int, str] = {
    0: "fixation",
    1: "rule1",
    2: "rule2",
    3: "rule3",
    4: "transition",
    5: "test1",
    6: "test2",
    7: "test3",
    8: "response",
}

# Event codes observed in the current dataset.
EXPECTED_EVENT_CODES_BY_POS: Dict[int, Tuple[int, ...]] = {
    0: (2,),  # fixation
    1: (5,),  # rule
    2: (5,),
    3: (5,),
    4: (9,),  # transition
    5: (6,),  # test
    6: (6,),
    7: (7, 8),  # test_match / test_non_match
    8: (4,),  # response
}


@dataclass(frozen=True)
class DatasetIndex:
    n_trials: int
    n_times: int
    times_s: np.ndarray  # (n_times,)
    trial_ids: np.ndarray  # (n_epochs,)
    epoch_pos: np.ndarray  # (n_epochs,)


def _validate_and_index_trials(event_codes: np.ndarray, *, chunk: int = 9) -> DatasetIndex:
    n_epochs = int(event_codes.shape[0])
    if n_epochs % chunk != 0:
        raise ValueError(f"Expected n_epochs divisible by {chunk}; got {n_epochs}.")

    n_trials = n_epochs // chunk

    # Validate repeating pattern.
    for trial in range(n_trials):
        start = trial * chunk
        seq = event_codes[start : start + chunk]
        for pos in range(chunk):
            allowed = EXPECTED_EVENT_CODES_BY_POS[pos]
            if int(seq[pos]) not in allowed:
                raise ValueError(
                    "Unexpected event-code pattern. "
                    f"Trial={trial}, pos={pos}, code={int(seq[pos])}, allowed={allowed}."
                )

    # Important: do NOT create the arange in int8 (it overflows for n_epochs > 127).
    ar = np.arange(n_epochs, dtype=np.int32)
    trial_ids = ar // chunk
    epoch_pos = (ar % chunk).astype(np.int8)

    # Placeholder; filled by caller.
    return DatasetIndex(
        n_trials=n_trials,
        n_times=-1,
        times_s=np.array([], dtype=float),
        trial_ids=trial_ids,
        epoch_pos=epoch_pos,
    )


def _split_sequence_col(series: pd.Series, *, expected_len: int, colname: str) -> List[List[str]]:
    seq = series.fillna("").astype(str).str.split(",")
    seq_list: List[List[str]] = seq.tolist()
    bad = [i for i, items in enumerate(seq_list) if len(items) != expected_len]
    if bad:
        example_i = bad[0]
        raise ValueError(
            f"Column {colname!r} has rows not of length {expected_len}. "
            f"Example row {example_i}: {seq_list[example_i]!r}"
        )
    return seq_list


def _build_trial_labels(df_beh: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return (rule_type, stim_name_to_id).

    rule_type is (n_trials,) strings (e.g., ABA/ABB).
    stim_name_to_id maps stimulus names to 1..K.
    """
    if "trial" not in df_beh.columns:
        raise ValueError("Behavior file must include a 'trial' column.")

    df_beh = df_beh.sort_values("trial").reset_index(drop=True)

    if "rule_type" not in df_beh.columns:
        raise ValueError("Behavior file must include a 'rule_type' column (e.g., ABA/ABB).")

    rule_type = df_beh["rule_type"].astype(str).to_numpy()

    # Build a stable mapping to 1..K for stimulus identities.
    stim_cols = [c for c in ["A_stim", "B_stim", "X_stim", "Y_stim"] if c in df_beh.columns]
    stim_names: List[str] = []
    for c in stim_cols:
        stim_names.extend(df_beh[c].dropna().astype(str).tolist())

    # Also include any names in the sequences.
    if "rule_sequence" in df_beh.columns:
        for items in _split_sequence_col(df_beh["rule_sequence"], expected_len=3, colname="rule_sequence"):
            stim_names.extend(items)
    if "test_sequence" in df_beh.columns:
        for items in _split_sequence_col(df_beh["test_sequence"], expected_len=3, colname="test_sequence"):
            stim_names.extend(items)

    stim_names = [s.strip() for s in stim_names if s.strip()]
    unique = sorted(set(stim_names))

    stim_name_to_id = {name: i + 1 for i, name in enumerate(unique)}
    return rule_type, stim_name_to_id


def _stimulus_id_for_epoch(
    df_beh: pd.DataFrame,
    stim_name_to_id: Dict[str, int],
    *,
    epoch_name: str,
) -> np.ndarray:
    df_beh = df_beh.sort_values("trial").reset_index(drop=True)

    rule_seq = _split_sequence_col(df_beh["rule_sequence"], expected_len=3, colname="rule_sequence")
    test_seq = _split_sequence_col(df_beh["test_sequence"], expected_len=3, colname="test_sequence")

    if epoch_name == "rule1":
        names = [items[0] for items in rule_seq]
    elif epoch_name == "rule2":
        names = [items[1] for items in rule_seq]
    elif epoch_name == "rule3":
        names = [items[2] for items in rule_seq]
    elif epoch_name == "test1":
        names = [items[0] for items in test_seq]
    elif epoch_name == "test2":
        names = [items[1] for items in test_seq]
    elif epoch_name == "test3":
        names = [items[2] for items in test_seq]
    else:
        names = [""] * len(df_beh)

    ids = np.array([stim_name_to_id.get(str(n).strip(), -1) for n in names], dtype=np.int16)
    ids[ids <= 0] = -1
    return ids


@dataclass
class PCAProjection:
    # (n_trials, n_times, 3)
    coords: np.ndarray
    axis_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def _fit_pca_and_project(
    X_epochs: np.ndarray,  # (n_trials, n_ch, n_times)
    *,
    random_state: int = 0,
) -> PCAProjection:
    from sklearn.decomposition import PCA

    n_trials, n_ch, n_times = X_epochs.shape

    # Fit PCA on all timepoints within this epoch across trials.
    X_fit = X_epochs.transpose(0, 2, 1).reshape(n_trials * n_times, n_ch).astype(np.float32, copy=False)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=random_state)
    coords = pca.fit_transform(X_fit).astype(np.float32, copy=False)

    coords = coords.reshape(n_trials, n_times, 3)

    mins = coords.reshape(-1, 3).min(axis=0)
    maxs = coords.reshape(-1, 3).max(axis=0)

    # Small padding so points don't sit on axes.
    pad = 0.05 * (maxs - mins + 1e-9)
    mins = mins - pad
    maxs = maxs + pad

    axis_ranges = ((float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1])), (float(mins[2]), float(maxs[2])))

    return PCAProjection(coords=coords, axis_ranges=axis_ranges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        required=True,
        help="Path to MNE epochs .fif (e.g., tim/ported_results/sub-001_ses-01_task-binding_epo.fif)",
    )
    parser.add_argument(
        "--behavior",
        required=True,
        help="Path to behavioral trial table CSV (e.g., data/2026-04-10/sub-001_events.csv)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    import mne

    mne.set_log_level("ERROR")

    epochs = mne.read_epochs(args.epochs, preload=True, verbose="ERROR")
    data = epochs.get_data(picks="meg")  # (n_epochs, n_ch, n_times)

    index = _validate_and_index_trials(epochs.events[:, 2], chunk=9)
    index = DatasetIndex(
        n_trials=index.n_trials,
        n_times=int(data.shape[2]),
        times_s=epochs.times.copy(),
        trial_ids=index.trial_ids,
        epoch_pos=index.epoch_pos,
    )

    df_beh = pd.read_csv(args.behavior)
    if len(df_beh) != index.n_trials:
        raise ValueError(
            f"Behavior rows ({len(df_beh)}) must match inferred n_trials ({index.n_trials})."
        )

    rule_type, stim_name_to_id = _build_trial_labels(df_beh)

    # Build per-epoch stimulus IDs (n_trials,)
    stim_ids_by_epoch: Dict[str, np.ndarray] = {
        name: _stimulus_id_for_epoch(df_beh, stim_name_to_id, epoch_name=name)
        for name in set(EPOCH_POS_TO_NAME.values())
    }

    # Build indices selecting the single epoch instance for each trial.
    # For each epoch position, there is exactly one epoch per trial.
    indices_by_epoch: Dict[str, np.ndarray] = {}
    for pos, name in EPOCH_POS_TO_NAME.items():
        idx = np.where(index.epoch_pos == pos)[0]
        if idx.shape[0] != index.n_trials:
            raise ValueError(f"Expected {index.n_trials} epochs for {name}, got {idx.shape[0]}")
        indices_by_epoch[name] = idx

    # Fit PCA per within-trial epoch and precompute coordinates for all times.
    proj_by_epoch: Dict[str, PCAProjection] = {}
    for name, idx in indices_by_epoch.items():
        X = data[idx, :, :]
        proj_by_epoch[name] = _fit_pca_and_project(X)

    # Dash app
    from dash import Dash, Input, Output, dcc, html
    import plotly.express as px

    n_times = index.n_times
    times_ms = (index.times_s * 1000.0).round(1)

    marks = {
        0: f"{times_ms[0]} ms",
        int(n_times // 2): f"{times_ms[int(n_times // 2)]} ms",
        n_times - 1: f"{times_ms[-1]} ms",
    }

    epoch_options = [{"label": name, "value": name} for name in EPOCH_POS_TO_NAME.values()]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("PCA Explorer (MEG epochs)", style={"marginBottom": "0.25rem"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("PCA space / epoch"),
                            dcc.Dropdown(
                                id="epoch_name",
                                options=epoch_options,
                                value="rule1",
                                clearable=False,
                            ),
                        ],
                        style={"width": "260px", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        [
                            html.Label("Coloring"),
                            dcc.RadioItems(
                                id="color_scheme",
                                options=[
                                    {"label": "Stimulus (1–5)", "value": "stimulus"},
                                    {"label": "Rule (ABA/ABB)", "value": "rule"},
                                ],
                                value="stimulus",
                            ),
                        ],
                        style={"marginLeft": "24px", "display": "inline-block", "verticalAlign": "top"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Time within epoch"),
                    dcc.Slider(
                        id="time_idx",
                        min=0,
                        max=n_times - 1,
                        step=1,
                        value=int(np.argmin(np.abs(index.times_s - 0.0))),
                        marks=marks,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"marginTop": "12px", "marginBottom": "6px"},
            ),
            dcc.Graph(id="pca_plot", style={"height": "780px"}),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto"},
    )

    @app.callback(
        Output("pca_plot", "figure"),
        Input("epoch_name", "value"),
        Input("time_idx", "value"),
        Input("color_scheme", "value"),
    )
    def _update(epoch_name: str, time_idx: int, color_scheme: str):
        proj = proj_by_epoch[epoch_name]
        coords_t = proj.coords[:, int(time_idx), :]

        df = pd.DataFrame(
            {
                "PC1": coords_t[:, 0],
                "PC2": coords_t[:, 1],
                "PC3": coords_t[:, 2],
                "trial": np.arange(index.n_trials, dtype=int),
                "rule": rule_type,
            }
        )

        stim_ids = stim_ids_by_epoch.get(epoch_name)
        if stim_ids is None:
            df["stimulus"] = "(none)"
        else:
            df["stimulus"] = [str(x) if x > 0 else "(none)" for x in stim_ids]

        color_col = "stimulus" if color_scheme == "stimulus" else "rule"

        t_ms = float(times_ms[int(time_idx)])
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            color=color_col,
            hover_name="trial",
            title=f"{epoch_name} PCA space • t = {t_ms:.1f} ms",
        )

        (xr, yr, zr) = proj.axis_ranges
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            scene={
                "xaxis": {"title": "PC1", "range": list(xr)},
                "yaxis": {"title": "PC2", "range": list(yr)},
                "zaxis": {"title": "PC3", "range": list(zr)},
            },
            legend_title_text=color_col,
        )

        return fig

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
