#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataLoader Module
-----------------
Une Module pour recuperer la dataset (csv,excel et json) et elle retourne le data frame 
"""

import os
import pandas as pd
from typing import Optional

class DataLoader:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.dataset = None

    def load_dataset(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset (CSV, Excel, JSON)."""
        path = path or self.path
        if not path:
            raise ValueError("A dataset path must be provided.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Use CSV, Excel, or JSON.")
        self.dataset = df
        return df

    def get_columns(self):
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded yet.")
        return self.dataset.columns.tolist()

    def shape(self):
        if self.dataset is None:
            return (0, 0)
        return self.dataset.shape
