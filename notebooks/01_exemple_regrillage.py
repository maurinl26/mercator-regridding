"""
Exemple de regrillage NEMO/ORCA025 -> grille standard.

Exécution :

    uv run python notebooks/01_exemple_regrillage.py

Déroulé :
1. Charger le mesh_mask (grille curvilinéaire ORCA025, mer du Nord).
2. Charger les champs de vitesse de surface natifs (uos, vos).
3. Construire une grille cible standard régulière.
4. Regriller le scalaire (uos seul, pour valider la mécanique).
5. Regriller le vecteur (uos, vos) -- sans angle de rotation ici, cf. note.
6. Comparer visuellement (avant/après) et sauvegarder les figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from regridding.core import build_target_grid, regrid_scalar, regrid_vector

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "NemoNorthSeaORCA025-N006_data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def load_mesh_mask() -> xr.Dataset:
    """Charge le mesh_mask (coordonnées glamf/gphif de la grille native)."""
    path = DATA_DIR / "coordinates.nc"
    return xr.open_dataset(path, decode_times=False)


def load_velocity_fields() -> xr.Dataset:
    """Charge les champs de vitesse de surface natifs (uos, vos)."""
    ds_u = xr.open_dataset(DATA_DIR / "ORCA025-N06_20000104d05U.nc", decode_times=False)
    ds_v = xr.open_dataset(DATA_DIR / "ORCA025-N06_20000104d05V.nc", decode_times=False)
    return xr.merge([ds_u[["uos"]], ds_v[["vos"]]], compat="override", join="override")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh = load_mesh_mask()
    print("Mesh mask :", list(mesh.data_vars))
    print("Dimensions grille native :", mesh.sizes)

    velocities = load_velocity_fields()
    print("Champs de vitesse (surface) :", list(velocities.data_vars))

    # Grille native : coordonnées 2D (curvilinéaire).
    lon_in = mesh["glamf"].isel(time=0).values
    lat_in = mesh["gphif"].isel(time=0).values

    # Champs au premier (et seul) pas de temps.
    uos = velocities["uos"].isel(time_counter=0).values
    vos = velocities["vos"].isel(time_counter=0).values

    print(f"Grille native : {lon_in.shape}, emprise lon [{lon_in.min():.2f}, "
          f"{lon_in.max():.2f}], lat [{lat_in.min():.2f}, {lat_in.max():.2f}]")

    # Grille cible standard, calée sur l'emprise de la zone mer du Nord.
    lon_out, lat_out = build_target_grid(
        lon_min=float(lon_in.min()),
        lon_max=float(lon_in.max()),
        lat_min=float(lat_in.min()),
        lat_max=float(lat_in.max()),
        resolution_deg=0.1,
    )
    print(f"Grille cible standard : {lon_out.shape}, résolution 0.1°")

    # --- 1. Regrillage scalaire (uos seul), pour valider la mécanique ---
    result_scalar = regrid_scalar(
        lon_in, lat_in, uos, lon_out, lat_out, method="gauss"
    )
    print(f"Regrillage scalaire (uos) : méthode={result_scalar.method}, "
          f"shape sortie={result_scalar.data.shape}")

    # --- 2. Regrillage vectoriel (uos, vos) ---
    # NOTE : ce mesh_mask simplifié ne fournit pas d'angle de rotation
    # explicite (variable "angle" absente). Étape suivante pour un vrai
    # rendu Mercator : calculer l'angle à partir des dérivées locales de
    # glamu/gphiu (cf. doc NEMO / mesh_mask complet), ou l'extraire d'un
    # mesh_mask non tronqué. Ici, on documente l'hypothèse simplificatrice
    # (rotation nulle) plutôt que de la cacher.
    u_result, v_result = regrid_vector(
        lon_in, lat_in, uos, vos, lon_out, lat_out, angle=None, method="gauss"
    )
    print("Regrillage vectoriel (uos, vos) : OK (rotation non appliquée, "
          "angle absent du mesh_mask -- limite documentée du jeu de test)")

    # --- 3. Comparaison visuelle avant/après ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].pcolormesh(lon_in, lat_in, uos, shading="auto", cmap="RdBu_r")
    axes[0].set_title("uos -- grille native ORCA025 (curvilinéaire)")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    fig.colorbar(im0, ax=axes[0], label="m/s")

    im1 = axes[1].pcolormesh(
        lon_out, lat_out, result_scalar.data, shading="auto", cmap="RdBu_r"
    )
    axes[1].set_title("uos -- grille standard (0.1°, regrillage gaussien)")
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel("latitude")
    fig.colorbar(im1, ax=axes[1], label="m/s")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "comparaison_avant_apres.png"
    fig.savefig(out_path, dpi=120)
    print(f"Figure sauvegardée : {out_path}")

    # --- 4. Métrique de contrôle : la moyenne globale doit être proche ---
    mean_native = float(np.nanmean(uos))
    mean_regrid = float(np.nanmean(result_scalar.data))
    print(f"Moyenne uos native : {mean_native:.4f} m/s")
    print(f"Moyenne uos regrillée : {mean_regrid:.4f} m/s")
    print(f"Écart relatif : {abs(mean_native - mean_regrid) / abs(mean_native) * 100:.2f} %")


if __name__ == "__main__":
    main()
