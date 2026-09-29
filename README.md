# Mercator Regridding — exemple pédagogique (Ophélie)

Petit exercice pour évaluer le regrillage de champs océanographiques sur grille
native NEMO/ORCA (curvilinéaire, C-grid Arakawa) vers une grille standard —
même problématique que le DCE Mercator Ocean « Cloud Optimised Regridding »
(24249L00).

## Données

Jeu d'exemple public NEMO/ORCA025, mer du Nord (`NemoNorthSeaORCA025-N006_data`),
téléchargé via le paquet `parcels` (OceanParcels) — contient :

- `coordinates.nc` (mesh_mask) : `glamf`/`gphif` (lon/lat aux coins des cellules,
  grille curvilinéaire), aires de cellules.
- `ORCA*U.nc`, `ORCA*V.nc`, `ORCA*W.nc` : champs de vitesse sur grille C
  (U et V pas au même nœud — le vrai test de rotation vectorielle).

Le Copernicus Marine Data Store ne livre JAMAIS la grille native (uniquement
la grille standard Arakawa-A déjà interpolée) — d'où le choix de ce jeu externe
qui reproduit fidèlement le cas natif visé par le marché.

## Objectif de l'exercice

1. Charger la grille source curvilinéaire (ORCA025) et construire une grille
   cible standard régulière (ex. 0.05°).
2. Regriller un champ scalaire (bilinéaire ou conservatif).
3. Regriller le champ vectoriel U/V **avec rotation des composantes** —
   le point le plus discriminant du marché.
4. Comparer visuellement (cartopy) et quantitativement (RMSE, conservation
   de flux) avant/après.
5. Organiser le code comme un embryon du service demandé par le DCE :
   une fonction `regrid(ds_in, ds_out, method, variable_type)` réutilisable.

## Structure

```
mercator-regridding-exemple/
├── data/               # données téléchargées (gitignored)
├── scripts/
│   └── download_data.py
├── notebooks/
├── src/
│   └── regridding/
├── pyproject.toml
└── README.md
```

## Installation

```bash
uv sync
```
# Mercator Regriding examples
