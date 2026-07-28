"""
lf_brainview.py — lightweight glass-brain electrode renders on fsaverage.

The 252 recon path renders with PyVista (photoreal, one PNG per cluster x view x
colouring) but PyVista is not installed everywhere and each render is expensive.
This module gives a fast, dependency-light alternative for FIGURES: project the
fsaverage pial surface to a soft silhouette with matplotlib and scatter the
electrodes on top, in the standard three views.

Coordinates are fsaverage (MNI305) — the same space the recon coords CSVs and the
surface .gii meshes use, so no transform is needed.

    render_groups(df, groups, out_png=...)   # categorical: in/out, cluster id, ...
    render_scalar(df, values, out_png=...)   # continuous: atlas probability, ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
MESH_DIR = _REPO / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "meshes"

# (label, axis pair, which hemisphere's silhouette, flip-x) for each view
VIEWS = {
    "lateral_L": dict(ax=(1, 2), hemi="lh", invert_x=False, xlabel="y (post → ant)", ylabel="z"),
    "lateral_R": dict(ax=(1, 2), hemi="rh", invert_x=True,  xlabel="y (ant ← post)", ylabel="z"),
    "dorsal":    dict(ax=(0, 1), hemi="both", invert_x=False, xlabel="x (L → R)",    ylabel="y"),
}

_MESH_CACHE: dict = {}


def _load_surface(hemi: str) -> np.ndarray:
    """(n_vertices, 3) fsaverage pial coordinates."""
    if hemi in _MESH_CACHE:
        return _MESH_CACHE[hemi]
    import nibabel as nib
    g = nib.load(str(MESH_DIR / f"fsaverage_{hemi}.gii"))
    verts = np.asarray(g.agg_data()[0] if isinstance(g.agg_data(), tuple) else g.agg_data()[0],
                       dtype=float)
    _MESH_CACHE[hemi] = verts
    return verts


def _silhouette(ax, hemi: str, a0: int, a1: int, *, n: int = 9000, seed: int = 0):
    """Soft grey brain outline: a subsampled vertex cloud in the projection plane."""
    parts = ["lh", "rh"] if hemi == "both" else [hemi]
    V = np.vstack([_load_surface(h) for h in parts])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=min(n, len(V)), replace=False)
    ax.scatter(V[idx, a0], V[idx, a1], s=3.0, c="#c9ccd1", alpha=0.16,
               linewidths=0, rasterized=True, zorder=0)


def _view_mask(df, view: str) -> np.ndarray:
    """Lateral views show only that hemisphere's electrodes — otherwise the two
    laterals are the same cloud mirrored, which reads as a real difference and is not.
    Uses the `hemi` column when present, else the sign of x."""
    n = len(df)
    if view == "dorsal":
        return np.ones(n, dtype=bool)
    want = "L" if view.endswith("_L") else "R"
    if "hemi" in getattr(df, "columns", []):
        h = df["hemi"].astype(str).str.upper().str[0].to_numpy()
        if set(np.unique(h)) & {"L", "R"}:
            return h == want
    x = df["x"].to_numpy(dtype=float)
    return x < 0 if want == "L" else x >= 0


def _frame(ax, view: str, cfg: dict):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if cfg["invert_x"]:
        ax.invert_xaxis()
    ax.set_title(view.replace("_", " "), fontsize=9, color="#555")


def render_groups(df, groups, *, colors: Dict[str, str], out_png,
                  title: str = "", views: Sequence[str] = ("lateral_L", "lateral_R", "dorsal"),
                  order: Optional[Sequence[str]] = None, sizes: Optional[Dict[str, float]] = None,
                  legend: bool = True, dpi: int = 130, figsize_per: float = 3.6):
    """Scatter electrodes coloured by a categorical `groups` array.
    df needs x / y / z columns (fsaverage). `order` controls draw order (last on top)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(groups).astype(str)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    keys = list(order) if order else sorted(set(g))
    sizes = sizes or {}

    fig, axes = plt.subplots(1, len(views), figsize=(figsize_per * len(views), figsize_per + 0.5))
    axes = np.atleast_1d(axes)
    for ax, view in zip(axes, views):
        cfg = VIEWS[view]; a0, a1 = cfg["ax"]
        _silhouette(ax, cfg["hemi"], a0, a1)
        vm = _view_mask(df, view)
        for k in keys:
            m = (g == k) & vm
            if not m.any():
                continue
            ax.scatter(xyz[m, a0], xyz[m, a1], s=sizes.get(k, 11), c=colors.get(k, "#888"),
                       alpha=0.9, linewidths=0.3, edgecolors="white", zorder=2)
        _frame(ax, view, cfg)

    if legend:
        counts = {k: int((g == k).sum()) for k in keys}
        handles = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                              markerfacecolor=colors.get(k, "#888"), markeredgecolor="white",
                              label=f"{k}  (n={counts[k]})") for k in keys if counts[k]]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def render_scalar(df, values, *, out_png, title: str = "", cmap: str = "viridis",
                  views: Sequence[str] = ("lateral_L", "lateral_R", "dorsal"),
                  vmin: Optional[float] = None, vmax: Optional[float] = None,
                  label: str = "", dpi: int = 130, figsize_per: float = 3.6,
                  zero_color: str = "#e3e5e8"):
    """Scatter electrodes coloured by a continuous value (e.g. atlas probability).
    Contacts at (or below) vmin are drawn in `zero_color` so the network stands out."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(values, dtype=float)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    finite = np.isfinite(v)
    vmin = float(vmin if vmin is not None else 0.0)
    vmax = float(vmax if vmax is not None else np.nanpercentile(v[finite], 99))
    hot = finite & (v > vmin)

    fig, axes = plt.subplots(1, len(views), figsize=(figsize_per * len(views), figsize_per + 0.5))
    axes = np.atleast_1d(axes)
    sc = None
    for ax, view in zip(axes, views):
        cfg = VIEWS[view]; a0, a1 = cfg["ax"]
        _silhouette(ax, cfg["hemi"], a0, a1)
        vm = _view_mask(df, view)
        cold = finite & ~hot & vm
        hot_v = hot & vm
        ax.scatter(xyz[cold, a0], xyz[cold, a1], s=7, c=zero_color, alpha=0.7,
                   linewidths=0, zorder=1)
        o = np.argsort(v[hot_v])                     # strongest drawn last
        sc = ax.scatter(xyz[hot_v, a0][o], xyz[hot_v, a1][o], s=18, c=v[hot_v][o], cmap=cmap,
                        vmin=vmin, vmax=vmax, alpha=0.95, linewidths=0.3,
                        edgecolors="white", zorder=2)
        _frame(ax, view, cfg)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cb.set_label(label or "value", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def render_thumb(df, mask, *, out_png, color: str = "#1f77b4", dpi: int = 110,
                 view: str = "lateral_L", size: float = 1.9):
    """Small single-view chip: highlighted subset on a faint all-electrode backdrop.
    Sized to sit beside a cluster-centroid thumbnail."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    m = np.asarray(mask, dtype=bool)
    cfg = VIEWS[view]; a0, a1 = cfg["ax"]
    fig, ax = plt.subplots(figsize=(size, size))
    _silhouette(ax, cfg["hemi"], a0, a1, n=5000)
    vm = _view_mask(df, view)
    bg, fg = (~m) & vm, m & vm
    ax.scatter(xyz[bg, a0], xyz[bg, a1], s=2.0, c="#d7dade", alpha=0.55, linewidths=0, zorder=1)
    ax.scatter(xyz[fg, a0], xyz[fg, a1], s=9, c=color, alpha=0.95, linewidths=0.2,
               edgecolors="white", zorder=2)
    _frame(ax, view, cfg); ax.set_title("")
    fig.tight_layout(pad=0.05)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    return out_png
