# core/result_schema.py
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class JobResult:
    task_type: str                  # "classification" | "regression"
    summary: str                    # texte synthétique
    table: pd.DataFrame             # résultats tabulaires
    logs: str                       # logs capturés
    error: Optional[str] = None     # message d'erreur si crash
