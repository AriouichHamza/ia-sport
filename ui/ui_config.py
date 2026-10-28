# ui/ui_config.py
import os

DEFAULT_SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

APP_TITLE = "AISport – POC (Classification / Regression)"
APP_DESCRIPTION = (
    "Interface Gradio pour exécuter les pipelines AISport sur un dataset (CSV), "
    "lancer contextualisation + entraînement + comparaison modèles, et afficher résultats."
)
