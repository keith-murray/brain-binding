"""
Sensor-space plotting utilities.

Two entry points:
  plot_sensor_map(values, pos)        — 2D topomap, one value per sensor
  plot_sensor_map_3d(matrix, pos)     — 3D stacked topomap, (n_sensors, n_times)

Helper:
  get_sensor_positions(info)          — extract 2-D (x, y) from an mne.Info object

The dataset has 64 physical locations × 3 axes = 192 channels.
By default get_sensor_positions returns the Z-axis (radial) channels only,
giving 64 positions that map cleanly to a 2-D head layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Position helper
# ---------------------------------------------------------------------------

def get_sensor_positions(
    info,
    axis: str = "Z",
) -> Tuple[np.ndarray, list]:
    """Return 2-D sensor positions from an MNE Info object.

    Parameters
    ----------
    info : mne.Info
    axis : 'Z' | 'Y' | 'X' | None
        Which axis suffix to select. None returns all 192 channels.

    Returns
    -------
    pos   : (n_sensors, 2) float array, x/y in [0, 1]
    names : list of channel name strings
    """
    from mne.channels import find_layout

    layout = find_layout(info, ch_type="mag")
    if axis is not None:
        idx = [i for i, n in enumerate(layout.names) if n.endswith(f" {axis}")]
    else:
        idx = list(range(len(layout.names)))

    pos   = layout.pos[idx, :2].copy()
    names = [layout.names[i] for i in idx]
    return pos, names


# ---------------------------------------------------------------------------
# 2-D topomap
# ---------------------------------------------------------------------------

def plot_sensor_map(
    values: np.ndarray,
    pos: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    cmap: str = "Reds",
    alpha: float = 0.3,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
) -> plt.Axes:
    """2-D sensor-space heatmap via Delaunay triangulation.

    Parameters
    ----------
    values : (n_sensors,) array — one scalar per sensor
    pos    : (n_sensors, 2) array — x/y positions from get_sensor_positions()
    ax     : existing Axes, or None to create one
    title  : axes title string
    cmap   : matplotlib colormap name
    alpha  : mesh transparency
    vmin, vmax : colour scale limits; defaults to data min/max

    Returns
    -------
    ax : the Axes with the plot
    """
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) == len(pos), (
        f"values length {len(values)} != pos rows {len(pos)}"
    )

    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    x, y = pos[:, 0], pos[:, 1]
    tri = mtri.Triangulation(x, y)

    # Filled contour mesh
    tcf = ax.tripcolor(tri, values, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                       shading="gouraud")

    # Sensor dots
    ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
               s=20, zorder=3, linewidths=0)

    if colorbar:
        plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# 3-D stacked topomap
# ---------------------------------------------------------------------------

def plot_sensor_map_3d(
    matrix: np.ndarray,
    pos: np.ndarray,
    times: Optional[np.ndarray] = None,
    ax=None,
    title: str = "",
    cmap: str = "Reds",
    alpha: float = 0.3,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    n_time_ticks: int = 5,
):
    """3-D sensor heatmap: sensor plane (x, y) × time (z).

    Each time slice is rendered as a triangulated mesh at z = times[t],
    face-coloured by the corresponding sensor values.

    Parameters
    ----------
    matrix : (n_sensors, n_times) array
    pos    : (n_sensors, 2) from get_sensor_positions()
    times  : (n_times,) time axis values in seconds; defaults to 0..n_times-1
    ax     : existing Axes3D, or None to create one
    title  : figure title
    cmap   : matplotlib colormap name
    alpha  : surface transparency
    vmin, vmax : colour scale limits; defaults to global data min/max

    Returns
    -------
    ax : the Axes3D
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    matrix = np.asarray(matrix, dtype=float)
    assert matrix.ndim == 2 and matrix.shape[0] == len(pos), (
        f"matrix.shape[0] {matrix.shape[0]} != n_sensors {len(pos)}"
    )
    n_sensors, n_times = matrix.shape

    if times is None:
        times = np.arange(n_times, dtype=float)

    vmin = float(matrix.min()) if vmin is None else vmin
    vmax = float(matrix.max()) if vmax is None else vmax

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax  = fig.add_subplot(111, projection="3d")

    x, y = pos[:, 0], pos[:, 1]
    tri  = mtri.Triangulation(x, y)

    norm    = Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    # Per-face colours: average the three vertex values for each triangle
    # tri.triangles: (n_faces, 3) vertex indices
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for t_idx, t_val in enumerate(times):
        vals       = matrix[:, t_idx]                              # (n_sensors,)
        face_vals  = vals[tri.triangles].mean(axis=1)              # (n_faces,)
        face_colors = cmap_fn(norm(face_vals))                     # (n_faces, 4) RGBA

        # Build a list of (3, 3) vertex arrays: [x, y, z] per vertex per triangle
        verts = np.stack([
            x[tri.triangles],                                      # (n_faces, 3)
            y[tri.triangles],                                      # (n_faces, 3)
            np.full_like(x[tri.triangles], t_val),                 # (n_faces, 3)
        ], axis=-1)                                                # (n_faces, 3, 3)

        poly = Poly3DCollection(verts, alpha=alpha, shade=False)
        poly.set_facecolor(face_colors)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)

    # Colour bar via a dummy ScalarMappable
    if colorbar:
        sm = ScalarMappable(cmap=cmap_fn, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.6)

    # Tidy z axis ticks
    tick_idx = np.linspace(0, n_times - 1, n_time_ticks, dtype=int)
    ax.set_zticks(times[tick_idx])
    ax.set_zticklabels([f"{times[i]:.2f}" for i in tick_idx])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time (s)")
    ax.set_title(title)

    return ax
