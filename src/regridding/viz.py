"""Outils de tracé des maillages (grille native curvilinéaire vs grille cible)."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection


def mesh_lines(lon2d: np.ndarray, lat2d: np.ndarray, step: int = 1, bbox=None):
    """Segments des lignes i=cste et j=cste d'une grille 2D (curvilinéaire ou non).

    step : on ne trace qu'une ligne sur `step`.
    bbox : (lon_min, lon_max, lat_min, lat_max) pour ne garder que la zone utile.
    """
    lon = np.asarray(lon2d)
    lat = np.asarray(lat2d)
    if bbox is not None:
        lo0, lo1, la0, la1 = bbox
        inside = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    else:
        inside = np.ones(lon.shape, dtype=bool)

    segs = []
    ny, nx = lon.shape
    for j in range(0, ny, step):
        row = np.column_stack([lon[j], lat[j]])
        segs.extend(_split(row, inside[j]))
    for i in range(0, nx, step):
        col = np.column_stack([lon[:, i], lat[:, i]])
        segs.extend(_split(col, inside[:, i]))
    return segs


def _split(line: np.ndarray, keep: np.ndarray):
    """Découpe une polyligne en tronçons contigus de points conservés."""
    out, start = [], None
    for k, ok in enumerate(keep):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if k - start > 1:
                out.append(line[start:k])
            start = None
    if start is not None and len(keep) - start > 1:
        out.append(line[start:])
    return out


def draw_mesh(ax, lon2d, lat2d, step=1, bbox=None, color="k", lw=0.4, alpha=0.6, label=None):
    lc = LineCollection(mesh_lines(lon2d, lat2d, step, bbox), colors=color, linewidths=lw, alpha=alpha, label=label)
    ax.add_collection(lc)
    return lc
