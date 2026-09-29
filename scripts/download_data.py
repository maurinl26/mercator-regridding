"""Télécharge le jeu d'exemple NEMO/ORCA025 (mer du Nord) depuis le dépôt
`Parcels-code/parcels-data` (GitHub), en HTTP direct.

Ce jeu contient une vraie sortie NEMO sur grille native curvilinéaire ORCA025
(C-grid Arakawa) : mesh_mask (glamf/gphif) + champs U/V/W. C'est la structure
que le DCE Mercator vise (grille native NEMO/ORCA → grille standard CMEMS).

Le Copernicus Marine Data Store ne livre jamais la grille native — d'où le
choix de cette source externe, publique et légère.

Note : l'ancienne API `parcels.download_example_dataset()` a été retirée en
Parcels v4. On télécharge donc directement les fichiers bruts (raw.
githubusercontent.com), sans dépendre de l'API du paquet `parcels`.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASE_URL = "https://raw.githubusercontent.com/Parcels-code/parcels-data/main/data"

# On ne télécharge qu'un seul pas de temps (U/V/W) + le mesh_mask : suffisant
# pour l'exercice de regrillage (pas besoin des ~7 pas de temps disponibles).
NORTH_SEA_FILES = [
    "coordinates.nc",
    "ORCA025-N06_20000104d05U.nc",
    "ORCA025-N06_20000104d05V.nc",
    "ORCA025-N06_20000104d05W.nc",
]

# Petit jeu pédagogique : champ purement zonal sur grille ORCA025
# (aqua-planète). Utile pour vérifier la mécanique de rotation vectorielle
# sur un cas simple avant le cas réel mer du Nord.
CURVILINEAR_FILES = [
    "mesh_mask.nc4",
    "U_purely_zonal-ORCA025_grid_U.nc4",
    "V_purely_zonal-ORCA025_grid_V.nc4",
]


def download_dataset(dataset_name: str, filenames: list[str]) -> Path:
    target_dir = DATA_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        dest = target_dir / filename
        if dest.exists():
            print(f"  - {filename} (déjà présent)")
            continue
        url = f"{BASE_URL}/{dataset_name}/{filename}"
        print(f"  - téléchargement {filename} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"    -> {dest} ({len(response.content) / 1e6:.1f} Mo)")

    return target_dir


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement vers {DATA_DIR} ...")

    print("NemoCurvilinear_data (cas simple, aqua-planète) :")
    download_dataset("NemoCurvilinear_data", CURVILINEAR_FILES)

    print("NemoNorthSeaORCA025-N006_data (cas réel, mer du Nord) :")
    download_dataset("NemoNorthSeaORCA025-N006_data", NORTH_SEA_FILES)

    print("Terminé.")


if __name__ == "__main__":
    main()
