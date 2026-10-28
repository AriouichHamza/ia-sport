# core/dataset_service.py
from pathlib import Path
import pandas as pd

from modules.DataLoader import DataLoader

def load_dataset(dataset_path: str) -> pd.DataFrame:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    loader = DataLoader(p)
    df = loader.load_dataset()
    return df
