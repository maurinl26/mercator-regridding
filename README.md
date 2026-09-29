# Mercator Regridding — exemple pédagogique

Petit exercice pour évaluer le regrillage de champs océanographiques sur
grille native NEMO/ORCA (curvilinéaire, C-grid Arakawa) vers une grille
standard — même problématique que le DCE Mercator Ocean « Cloud Optimised
Regridding » (24249L00).

Objectif : présenter le sujet à Ophélie avant attribution du marché, sans
entrer dans la science du regrillage (elle est côté UI/UX + MCP) — juste
montrer concrètement de quoi il s'agit : grille native tordue → grille
standard régulière, sur un champ scalaire et un champ vectoriel.

## Installation

Tout le projet est géré en `uv`, pas de conda/micromamba.

```bash
uv sync
```

Ceci installe : xarray, cartopy, pyresample, matplotlib, streamlit, plotly,
netCDF4, parcels, requests, dans un venv local `.venv/`.

## Données

Jeu d'exemple public NEMO/ORCA025, mer du Nord
(`NemoNorthSeaORCA025-N006_data`), et un cas simple aqua-planète
(`NemoCurvilinear_data`), téléchargés depuis le dépôt GitHub
`Parcels-code/parcels-data` (HTTP direct — l'ancienne API
`parcels.download_example_dataset()` a été retirée en Parcels v4).

Contenu :
- `coordinates.nc` (mesh_mask) : `glamf`/`gphif` (lon/lat aux coins des
  cellules, grille curvilinéaire), aires de cellules.
- `ORCA*U.nc`, `ORCA*V.nc`, `ORCA*W.nc` : champs de vitesse sur grille C
  (U et V pas au même nœud — le test de rotation vectorielle).

Le Copernicus Marine Data Store ne livre JAMAIS la grille native (uniquement
la grille standard Arakawa-A déjà interpolée) — d'où le choix de ce jeu
externe qui reproduit fidèlement le cas natif visé par le marché.

Télécharger :

```bash
uv run mercator-regridding-download
```

## Choix technique : pyresample plutôt que xESMF

xESMF (la référence pour ce type de regrillage, remapping conservatif par
aire) dépend d'ESMF/ESMPy, qui **n'est installable que via conda** (pas de
wheel pip) — incompatible avec un workflow 100% `uv`.

`pyresample` est utilisé à la place : 100% pip, gère nativement les grilles
2D curvilinéaires via un kd-tree (les nœuds de la grille source et de la
grille cible sont projetés sur une sphère unité en coordonnées 3D
cartésiennes, puis interrogés par plus-proche-voisin ou pondération
gaussienne). C'est suffisant pour illustrer la mécanique du problème, mais
ce n'est **pas un remapping conservatif** — un écart de quelques % sur la
moyenne globale du champ regrillé est attendu (lissage gaussien), pas un
bug. Le vrai moteur de calcul du service (si le marché est remporté) devra
trancher entre repasser par ESMF via un environnement conda dédié, ou
implémenter un calcul de poids par recouvrement d'aire.

## Exemple : regrillage scalaire + vectoriel

```bash
uv run mercator-regridding-example
```

Ce script :
1. charge le mesh_mask (grille native ORCA025, mer du Nord) ;
2. charge les champs de vitesse de surface natifs (`uos`, `vos`) ;
3. construit une grille cible standard régulière (0.1° par défaut) ;
4. regrille le champ scalaire (`uos` seul) — valide la mécanique ;
5. regrille le champ vectoriel (`uos`, `vos`) — la rotation grille →
   géographique n'est pas appliquée ici (le mesh_mask simplifié utilisé ne
   fournit pas l'angle de rotation local ; à calculer à partir des dérivées
   de `glamu`/`gphiu` sur un mesh_mask complet) ;
6. compare visuellement (figure sauvegardée dans `outputs/`) et
   quantitativement (écart de moyenne avant/après).

Résultat typique : les structures spatiales (courant côtier norvégien,
tourbillons en mer de Norvège) sont bien préservées après regrillage ; la
moyenne globale du champ diffère d'environ 25-30 % à cause du lissage
gaussien (pas un artefact géographique).

## App interactive (Streamlit)

```bash
uv run mercator-regridding-app
```

(wrapper de `streamlit run app/streamlit_app.py` ; les options Streamlit passent telles quelles, ex. `uv run mercator-regridding-app --server.port 8502 --server.headless true`).

Affiche côte à côte la grille native et la grille regrillée, avec des
sliders pour ajuster la résolution cible, le rayon d'influence et le sigma
du lissage gaussien, plus un champ vectoriel (quiver) et un contrôle
qualité (écart de moyenne). Les maillages (natif déformé, cible régulière) sont affichés sur les cartes, plus un zoom qui superpose les deux grilles (natif en bleu, cible en rouge) pour visualiser la transformation. Pensée pour être manipulée directement, sans
lire le code.

## Structure

```
mercator-regridding/
├── data/                    # données téléchargées (gitignored)
├── outputs/                 # figures générées (gitignored)
├── app/
│   └── streamlit_app.py     # app interactive
├── src/
│   └── regridding/
│       ├── core.py          # regrid_scalar, regrid_vector, build_target_grid
│       ├── app.py           # entry point Streamlit
│       ├── viz.py           # tracé des maillages
│       ├── download.py      # téléchargement des données d'exemple
│       └── example.py       # script CLI de bout en bout
├── pyproject.toml
└── README.md
```

## Pitfall rencontré : venv non portable

Si l'app plante avec une erreur du type `bad interpreter: .../.venv/bin/
python3: no such file or directory` après avoir déplacé le dossier du
projet : le `.venv` contient des chemins absolus dans ses shebangs
(`.venv/bin/streamlit`, etc.) et **ne survit pas à un déplacement**. Solution :

```bash
rm -rf .venv
uv sync
```

`uv.lock` garantit que la réinstallation est rapide (tout est en cache
local) et identique.

## Troubleshooting Streamlit

- `bad interpreter ... .venv/bin/python` : venv déplacé, voir ci-dessus.
- Page blanche au premier chargement : le regrillage à froid prend quelques secondes ; recharger la page. Le warning pyresample "more than 8 neighbours" est attendu (rayon d'influence large).
- Port occupé : `uv run mercator-regridding-app --server.port 8502`.
- Message "Données introuvables" : lancer `uv run mercator-regridding-download`.
