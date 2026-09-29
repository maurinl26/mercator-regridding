"""Fonctions de regrillage — prototype pour le DCE Mercator Ocean.

Objectif : reproduire, à petite échelle, le cœur technique demandé par le
marché « Cloud Optimised Regridding » (24249L00) :

- interpolation native (curvilinéaire ORCA / C-grid) -> standard (régulier)
- variables scalaires (température, salinité) : plus proche voisin / gaussien
- variables vectorielles (U/V) : interpolation + rotation des composantes
  vers le repère géographique (est/nord), car U et V sur grille C sont
  exprimés dans le repère local de la grille, pas dans le repère
  géographique (sauf près de l'équateur où la grille ORCA est alignée).

Choix technique : `pyresample` plutôt que `xesmf`. xESMF/ESMPy n'est
installable que via conda (pas de wheel pip) ; pyresample est 100% pip et
gère nativement les grilles 2D (curvilinéaires) via kd-tree. Il ne fait pas
de remapping conservatif "à la ESMF" — pour un vrai chantier Mercator il
faudra le mentionner comme limite assumée, ou repasser par ESMF via un
environnement conda dédié si le conservatif rigoureux est requis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_gauss, resample_nearest


@dataclass
class RegriddingResult:
    """Résultat d'un regrillage, avec quelques métriques de contrôle."""

    data: np.ndarray
    method: str
    rmse_vs_reference: float | None = None


def build_target_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution_deg: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit une grille cible régulière (grille "standard" façon CMEMS).

    Retourne (lon2d, lat2d) — des tableaux 2D, format attendu par pyresample
    pour définir une AreaDefinition/SwathDefinition de sortie.
    """
    lon_1d = np.arange(lon_min, lon_max, resolution_deg)
    lat_1d = np.arange(lat_min, lat_max, resolution_deg)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return lon2d, lat2d


def _swath(lon2d: np.ndarray, lat2d: np.ndarray) -> SwathDefinition:
    return SwathDefinition(lons=lon2d, lats=lat2d)


def regrid_scalar(
    lon_in: np.ndarray,
    lat_in: np.ndarray,
    field: np.ndarray,
    lon_out: np.ndarray,
    lat_out: np.ndarray,
    method: str = "gauss",
    radius_of_influence: float = 50_000,
    sigma: float = 25_000,
) -> RegriddingResult:
    """Regrille un champ scalaire (température, salinité, ...).

    `lon_in`/`lat_in`/`field` : grille source curvilinéaire (2D) + valeurs.
    `lon_out`/`lat_out` : grille cible (2D, régulière ou non).

    Méthodes :
    - "nearest" : plus proche voisin (masques, données catégorielles).
    - "gauss" : pondération gaussienne (lisse, proche d'un bilinéaire en
      pratique pour des grilles de résolution comparable).
    """
    source = _swath(lon_in, lat_in)
    target = _swath(lon_out, lat_out)

    if method == "nearest":
        out = resample_nearest(
            source, field, target, radius_of_influence=radius_of_influence
        )
    elif method == "gauss":
        out = resample_gauss(
            source,
            field,
            target,
            radius_of_influence=radius_of_influence,
            sigmas=sigma,
            fill_value=None,  # hors domaine source : masqué (et non 0)
        )
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    return RegriddingResult(data=out, method=method)


def regrid_vector(
    lon_in: np.ndarray,
    lat_in: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lon_out: np.ndarray,
    lat_out: np.ndarray,
    angle: np.ndarray | None = None,
    method: str = "gauss",
    radius_of_influence: float = 50_000,
    sigma: float = 25_000,
) -> tuple[RegriddingResult, RegriddingResult]:
    """Regrille un champ vectoriel (U, V) avec rotation des composantes.

    Sur une grille curvilinéaire (ORCA/NEMO), U et V sont exprimés dans le
    repère local de la grille (souvent tourné par rapport au nord
    géographique, sauf près de l'équateur). Étapes :
      1. rotation (u, v) repère grille -> repère géographique (est/nord)
         à l'aide de l'angle de rotation local ;
      2. regrillage de u_east et v_north indépendamment (deviennent des
         scalaires au sens du regrillage) ;
      3. si la grille de sortie est elle-même tournée, refaire la rotation
         inverse après interpolation (non nécessaire ici : sortie régulière
         en repère est/nord natif).

    Si `angle` est None, on suppose (hypothèse à vérifier / documenter dans
    le rendu) que la zone est proche de l'équateur ou que le mesh_mask ne
    fournit pas cette info — étape à compléter en calculant l'angle à partir
    des dérivées locales de glamu/gphiu (cf. doc NEMO, `angle` dans les
    fichiers de grille étendus).
    """
    if angle is not None:
        u_east = u * np.cos(angle) - v * np.sin(angle)
        v_north = u * np.sin(angle) + v * np.cos(angle)
    else:
        u_east, v_north = u, v

    u_result = regrid_scalar(
        lon_in, lat_in, u_east, lon_out, lat_out, method, radius_of_influence, sigma
    )
    v_result = regrid_scalar(
        lon_in, lat_in, v_north, lon_out, lat_out, method, radius_of_influence, sigma
    )
    return u_result, v_result


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE simple entre deux champs déjà sur la même grille."""
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))
