"""App Streamlit — visualisation avant/après regrillage NEMO/ORCA025.

Lancer avec l'environnement uv du projet :

    uv run streamlit run app/streamlit_app.py

Présente, à titre d'exemple, le passage d'une grille native NEMO/ORCA
(curvilinéaire, C-grid) vers une grille standard régulière -- la
problématique centrale du DCE Mercator Ocean « Cloud Optimised Regridding »
(24249L00). Données : jeu d'exemple NEMO/ORCA025, mer du Nord (OceanParcels).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

from regridding.core import build_target_grid, regrid_scalar, regrid_vector

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "NemoNorthSeaORCA025-N006_data"

st.set_page_config(page_title="Regrillage Mercator — exemple", layout="wide")

st.title("Exemple de regrillage — grille native NEMO/ORCA025 → grille standard")
st.caption(
    "Exercice pédagogique pour le DCE Mercator Ocean « Cloud Optimised "
    "Regridding » (24249L00). Données : jeu d'exemple NEMO/ORCA025, mer du "
    "Nord (OceanParcels)."
)


@st.cache_data
def load_mesh_mask():
    path = DATA_DIR / "coordinates.nc"
    ds = xr.open_dataset(path, decode_times=False)
    lon = ds["glamf"].isel(time=0).values
    lat = ds["gphif"].isel(time=0).values
    return lon, lat


@st.cache_data
def load_velocity_fields():
    ds_u = xr.open_dataset(DATA_DIR / "ORCA025-N06_20000104d05U.nc", decode_times=False)
    ds_v = xr.open_dataset(DATA_DIR / "ORCA025-N06_20000104d05V.nc", decode_times=False)
    uos = ds_u["uos"].isel(time_counter=0).values
    vos = ds_v["vos"].isel(time_counter=0).values
    return uos, vos


@st.cache_data
def compute_regridding(resolution_deg: float, radius_km: float, sigma_km: float):
    lon_in, lat_in = load_mesh_mask()
    uos, vos = load_velocity_fields()

    lon_out, lat_out = build_target_grid(
        lon_min=float(lon_in.min()),
        lon_max=float(lon_in.max()),
        lat_min=float(lat_in.min()),
        lat_max=float(lat_in.max()),
        resolution_deg=resolution_deg,
    )

    result_scalar = regrid_scalar(
        lon_in, lat_in, uos, lon_out, lat_out,
        method="gauss",
        radius_of_influence=radius_km * 1000,
        sigma=sigma_km * 1000,
    )
    u_result, v_result = regrid_vector(
        lon_in, lat_in, uos, vos, lon_out, lat_out,
        angle=None,
        method="gauss",
        radius_of_influence=radius_km * 1000,
        sigma=sigma_km * 1000,
    )
    return lon_out, lat_out, result_scalar.data, u_result.data, v_result.data


try:
    lon_in, lat_in = load_mesh_mask()
    uos, vos = load_velocity_fields()
    data_available = True
except (FileNotFoundError, StopIteration):
    data_available = False
    st.warning(
        "Données introuvables. Lancer d'abord : "
        "`uv run python scripts/download_data.py`."
    )

if not data_available:
    st.stop()

st.sidebar.header("Paramètres du regrillage")
resolution = st.sidebar.slider("Résolution de la grille cible (°)", 0.02, 0.3, 0.1, 0.02)
radius_km = st.sidebar.slider("Rayon d'influence (km)", 10, 150, 50, 10)
sigma_km = st.sidebar.slider("Sigma (lissage gaussien, km)", 5, 100, 25, 5)

st.sidebar.info(
    "Méthode : pondération gaussienne (kd-tree, pyresample). Pas de "
    "remapping conservatif par aire (ESMF) dans cet exemple -- limite "
    "assumée, à traiter par le moteur de calcul final."
)

lon_out, lat_out, scalar_out, u_out, v_out = compute_regridding(
    resolution, radius_km, sigma_km
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Grille native (curvilinéaire ORCA025)")
    st.write(f"Dimensions : {lon_in.shape[0]} × {lon_in.shape[1]}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(lon_in, lat_in, uos, shading="auto", cmap="RdBu_r")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("uos — grille native (m/s)")
    fig.colorbar(im, ax=ax, label="m/s")
    st.pyplot(fig)

with col2:
    st.subheader("2. Grille standard (régulière)")
    st.write(f"Dimensions : {lon_out.shape[0]} × {lon_out.shape[1]}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(lon_out, lat_out, scalar_out, shading="auto", cmap="RdBu_r")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"uos — regrillé ({resolution}°, gauss)")
    fig.colorbar(im, ax=ax, label="m/s")
    st.pyplot(fig)

st.divider()
st.subheader("3. Champ vectoriel (U, V) regrillé")
st.write(
    "Rotation grille → géographique non appliquée (angle absent du "
    "mesh_mask simplifié utilisé pour cet exemple) — étape à compléter sur "
    "un mesh_mask complet."
)

fig, ax = plt.subplots(figsize=(8, 6))
step = max(1, lon_out.shape[0] // 40)
ax.quiver(
    lon_out[::step, ::step],
    lat_out[::step, ::step],
    u_out[::step, ::step],
    v_out[::step, ::step],
    scale=10,
)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Champ (U, V) regrillé, sous-échantillonné pour lisibilité")
st.pyplot(fig)

st.divider()
st.subheader("4. Contrôle qualité")
mean_native = float(np.nanmean(uos))
mean_regrid = float(np.nanmean(scalar_out))
ecart = abs(mean_native - mean_regrid) / abs(mean_native) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Moyenne uos native", f"{mean_native:.4f} m/s")
c2.metric("Moyenne uos regrillée", f"{mean_regrid:.4f} m/s")
c3.metric("Écart relatif", f"{ecart:.1f} %")

st.caption(
    "L'écart provient du lissage gaussien (kd-tree, pondération par "
    "distance) — pas d'un bug. Un remapping conservatif par aire (ESMF) "
    "réduirait cet écart ; c'est un choix à faire dans le moteur de calcul "
    "final du service, pas dans cet exemple pédagogique."
)
