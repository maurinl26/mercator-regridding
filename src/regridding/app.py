"""Point d'entrée pour lancer l'app Streamlit via `uv run mercator-regridding-app`.

`streamlit run` est une commande CLI qui lance son propre serveur (pas un
simple `main()` importable) -- ce petit wrapper permet de l'exposer comme un
entry point `[project.scripts]` normal, cohérent avec les autres commandes
du projet (`mercator-regridding-download`, `mercator-regridding-example`).
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


def main() -> None:
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
