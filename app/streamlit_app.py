"""App Streamlit — visualisation avant/après regrillage NEMO/ORCA025.

Lancer avec l'environnement uv du projet :

    uv run mercator-regridding-app

Prototype illustrant le passage d'une grille native NEMO/ORCA
(curvilinéaire, C-grid) vers une grille standard régulière -- la
problématique centrale du DCE Mercator Ocean « Cloud Optimised Regridding »
(24249L00). Données : jeu d'exemple NEMO/ORCA025, mer du Nord (OceanParcels).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import streamlit as st
import xarray as xr

from regridding.core import build_target_grid, regrid_scalar, regrid_vector
from regridding.viz import draw_mesh

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "NemoNorthSeaORCA025-N006_data"

st.set_page_config(page_title="Regrillage Mercator — prototype", layout="wide")

LAND_COLOR = "#d9cfae"


def plot_field(ax, lon, lat, field, land, outside=None, **kw):
    """Champ sur la mer ; terre en aplat beige ; trait de côte en noir."""
    sea = np.ma.masked_where(land | ~np.isfinite(np.ma.filled(field, np.nan)), field)
    ax.set_facecolor(LAND_COLOR)
    im = ax.pcolormesh(lon, lat, sea, shading="auto", cmap="RdBu_r", **kw)
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.9)
    if outside is not None and outside.any():  # hors emprise : blanc
        ax.pcolormesh(lon, lat, np.ma.masked_where(~outside, outside), shading="auto",
                      cmap=ListedColormap(["white"]))
    return im

st.title("Regrillage océanographique : de la grille native à la grille standard")
st.markdown(
    """
**Contexte.** Les modèles océaniques NEMO calculent sur une grille *native* (ORCA) :
curvilinéaire, déformée vers le pôle Nord, avec les variables (T, U, V) décalées
sur des nœuds différents (C-grid). Le service Copernicus Marine diffuse en revanche
des grilles *standard* régulières (latitude/longitude), plus simples à utiliser.
Le **regrillage** est la transformation de l'une à l'autre : chaque point de la grille
de destination est estimé à partir des points voisins de la grille source.

**Ce prototype** (pour le DCE Mercator Ocean « Cloud Optimised Regridding »,
24249L00) : champ de courant de surface NEMO/ORCA025, mer du Nord, regrillé vers
une grille régulière. Le service final devra couvrir d'autres couples de grilles,
méthodes et variables (scalaires, vecteurs, flux) ; ici, un seul cas est illustré.
"""
)

st.markdown(
    """
**Service cible** (cahier des charges, SOW 24249L00). Le regrillage sera un service
conteneurisé, sans état, déployé sur la plateforme **EDITO** (Digital Twin of the
Ocean) et publié dans son catalogue de services. Il doit tenir dans les quotas
standard d'une instance EDITO (8 CPU, 32 Go de RAM, sans GPU ; image < 1 Go). Les
données transitent par le stockage objet d'EDITO : l'utilisateur peut parcourir les
jeux de données du **Marine Data Store** de Copernicus Marine, importer des données
externes, et choisir la destination (par défaut son espace EDITO). Les sorties sont
au format NetCDF4 ou Zarr, conformes ARCO. Trois sens de regrillage sont attendus :
natif vers standard CMEMS, natif vers natif, standard vers natif.

Le service sera utilisable de deux façons :

- **Mode simple** : trois choix (source, méthode, destination), la méthode et ses
  paramètres étant sélectionnés automatiquement. L'utilisateur peut aussi lancer le
  regrillage en langage naturel via l'agent **Ocean Intelligence**, grâce à une
  interface **MCP** exposée par le service.
- **Mode avancé** : choix de la méthode et de ses paramètres, masques d'entrée et de
  sortie, personnalisation de la sortie (renommer des variables, retirer des dimensions).

*Ce prototype tourne en local : données d'exemple téléchargées, pas de
connexion au MDS ni à EDITO, pas d'interface MCP.*
"""
)
st.caption("Données : NEMO/ORCA025, mer du Nord (jeu public OceanParcels).")


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
def load_land_mask():
    """Masque terre (True) déduit des champs de vitesse : NaN ou vitesse nulle exacte."""
    uos, vos = load_velocity_fields()
    return np.isnan(uos) | np.isnan(vos) | ((uos == 0) & (vos == 0))


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
    land_out = regrid_scalar(
        lon_in, lat_in, load_land_mask().astype(float), lon_out, lat_out,
        method="gauss",
        radius_of_influence=radius_km * 1000,
        sigma=sigma_km * 1000,
    ).data
    outside_out = np.ma.getmaskarray(land_out)  # hors emprise de la grille native
    land_out = np.ma.filled(land_out, 0.0) >= 0.5
    return (lon_out, lat_out, result_scalar.data, u_result.data, v_result.data,
            land_out, outside_out)


def ensure_data():
    """Télécharge les fichiers mer du Nord s'ils manquent (ex. app déployée)."""
    needed = ["coordinates.nc", "ORCA025-N06_20000104d05U.nc", "ORCA025-N06_20000104d05V.nc"]
    if all((DATA_DIR / f).exists() for f in needed):
        return
    from regridding.download import download_dataset

    with st.spinner("Premier démarrage : téléchargement des données (~25 Mo)..."):
        download_dataset("NemoNorthSeaORCA025-N006_data", needed)


try:
    ensure_data()
    lon_in, lat_in = load_mesh_mask()
    uos, vos = load_velocity_fields()
    data_available = True
except Exception as exc:
    data_available = False
    st.error(
        f"Données indisponibles ({exc}). En local : "
        "`uv run mercator-regridding-download`."
    )

if not data_available:
    st.stop()

st.subheader("Définir le regrillage")
st.caption(
    "Mode simple : trois choix (source, méthode, destination). Les paramètres "
    "fins sont déterminés automatiquement ; ils sont modifiables en mode avancé."
)

c_src, c_meth, c_dst = st.columns(3)
with c_src:
    st.markdown("**1. Source**")
    st.selectbox(
        "Jeu de données",
        ["NEMO ORCA025 — mer du Nord (grille native, C-grid)"],
        help="Grille curvilinéaire : les lignes ne suivent pas les méridiens/parallèles.",
    )
    st.caption(f"Grille {lon_in.shape[0]} × {lon_in.shape[1]} — variables : uos, vos")
with c_meth:
    st.markdown("**2. Méthode**")
    st.selectbox(
        "Méthode d'interpolation",
        ["Automatique (pondération gaussienne)"],
        help="Autres méthodes du cahier des charges (bilinéaire, bicubique, "
        "conservatif ordre 1/2, IDW) : non implémentées dans ce prototype.",
    )
    st.caption("Pas de remapping conservatif par aire (ESMF) dans ce prototype : limite assumée.")
with c_dst:
    st.markdown("**3. Destination**")
    resolution = st.select_slider(
        "Grille standard (régulière), résolution (°)",
        options=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.3],
        value=0.1,
    )

with st.expander("Mode avancé : paramètres du lissage et affichage"):
    a1, a2, a3, a4 = st.columns(4)
    radius_km = a1.slider("Rayon d'influence (km)", 10, 150, 50, 10)
    sigma_km = a2.slider("Sigma (km)", 5, 100, 25, 5)
    show_mesh = a3.checkbox("Afficher les maillages", value=True)
    mesh_step = a4.slider("1 ligne de maillage sur N", 1, 20, 6)

lon_out, lat_out, scalar_out, u_out, v_out, land_out, outside_out = compute_regridding(
    resolution, radius_km, sigma_km
)

land_in = load_land_mask()
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Grille native (curvilinéaire ORCA025)")
    st.write(f"Dimensions : {lon_in.shape[0]} × {lon_in.shape[1]}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = plot_field(ax, lon_in, lat_in, uos, land_in)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if show_mesh:
        draw_mesh(ax, lon_in, lat_in, step=mesh_step, color="k", lw=0.3, alpha=0.5)
    ax.set_title("uos — grille native (m/s)")
    fig.colorbar(im, ax=ax, label="m/s")
    st.pyplot(fig)

with col2:
    st.subheader("2. Grille standard (régulière)")
    st.write(f"Dimensions : {lon_out.shape[0]} × {lon_out.shape[1]}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = plot_field(ax, lon_out, lat_out, scalar_out, land_out, outside_out)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if show_mesh:
        draw_mesh(ax, lon_out, lat_out, step=max(1, mesh_step * 3), color="k", lw=0.3, alpha=0.5)
    ax.set_title(f"uos — regrillé ({resolution}°, gauss)")
    fig.colorbar(im, ax=ax, label="m/s")
    st.pyplot(fig)

st.divider()
st.subheader("Zoom sur les maillages : native (bleu) vs cible (rouge)")
st.write(
    "Chaque point de la grille cible (rouge, régulière) est calculé à partir "
    "des points voisins de la grille native (bleue, déformée) : c'est ce "
    "passage qui est réalisé par le regrillage.\n\n"
    "**Méthode : kd-tree.** Les points de la grille native sont placés dans un "
    "kd-tree, construit sur leurs coordonnées cartésiennes 3D (sphère unité), ce "
    "qui évite les difficultés des longitudes/latitudes (pôle, changement de "
    "date). Pour chaque point cible, on cherche les voisins natifs dans le rayon "
    "d'influence, puis on les pondère par une gaussienne de la distance (sigma). "
    "Le trait de côte natif (noir) et celui de la grille cible (rouge pointillé) "
    "montrent l'effet du regrillage sur le masque terre/mer."
)
lo_min, lo_max = float(lon_in.min()), float(lon_in.max())
la_min, la_max = float(lat_in.min()), float(lat_in.max())
zc1, zc2, zc3 = st.columns(3)
zlon = zc1.slider("Longitude centre", lo_min, lo_max, 5.0)
zlat = zc2.slider("Latitude centre", la_min, la_max, 58.0)
zsize = zc3.slider("Taille de la fenêtre (°)", 1.0, 15.0, 4.0)
bbox = (zlon - zsize, zlon + zsize, zlat - zsize / 2, zlat + zsize / 2)

fig, ax = plt.subplots(figsize=(9, 5))
plot_field(ax, lon_in, lat_in, uos, land_in, alpha=0.35)
ax.contour(lon_out, lat_out, land_out.astype(float), levels=[0.5], colors="tab:red", linewidths=1.2, linestyles="--")
draw_mesh(ax, lon_in, lat_in, step=1, bbox=bbox, color="tab:blue", lw=0.8, alpha=0.9, label="native")
draw_mesh(ax, lon_out, lat_out, step=1, bbox=bbox, color="tab:red", lw=0.5, alpha=0.8, label="cible")
ax.set_xlim(bbox[0], bbox[1])
ax.set_ylim(bbox[2], bbox[3])
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.legend(loc="upper right")
st.pyplot(fig)

st.divider()
st.subheader("3. Champ vectoriel (U, V) regrillé")
st.write(
    "Rotation grille → géographique non appliquée (angle absent du "
    "mesh_mask simplifié utilisé pour ce prototype) — étape à compléter sur "
    "un mesh_mask complet."
)

fig, ax = plt.subplots(figsize=(8, 6))
step = max(1, lon_out.shape[0] // 40)
ax.quiver(
    lon_out[::step, ::step],
    lat_out[::step, ::step],
    np.ma.masked_where(land_out, u_out)[::step, ::step],
    np.ma.masked_where(land_out, v_out)[::step, ::step],
    scale=10,
)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_facecolor(LAND_COLOR)
ax.contour(lon_out, lat_out, land_out.astype(float), levels=[0.5], colors="k", linewidths=0.9)
ax.set_title("Champ (U, V) regrillé, sous-échantillonné pour lisibilité")
st.pyplot(fig)

st.divider()
st.subheader("4. Contrôle qualité")
mean_native = float(np.nanmean(np.where(land_in, np.nan, uos)))
mean_regrid = float(np.nanmean(np.where(land_out, np.nan, np.ma.filled(scalar_out, np.nan))))
ecart = abs(mean_native - mean_regrid) / abs(mean_native) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Moyenne uos native", f"{mean_native:.4f} m/s")
c2.metric("Moyenne uos regrillée", f"{mean_regrid:.4f} m/s")
c3.metric("Écart relatif", f"{ecart:.1f} %")

st.caption(
    "L'écart provient du lissage gaussien (kd-tree, pondération par "
    "distance) — pas d'un bug. Un remapping conservatif par aire (ESMF) "
    "réduirait cet écart ; c'est un choix à faire dans le moteur de calcul "
    "final du service, pas dans ce prototype."
)
