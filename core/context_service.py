# core/context_service.py
from typing import Tuple
import pandas as pd

from modules.ContextSctack import ContextSctack
from modules.Contextualization import Contextualization

def build_context(df: pd.DataFrame, user_prompt: str) -> Tuple[ContextSctack, str]:
    """
    Lance la contextualisation et retourne (context_stack, target_column).
    """
    context_stack = ContextSctack()
    contextualizer = Contextualization(
        dataframe=df,
        user_prompt=user_prompt,
        context_stack=context_stack
    )
    contextualizer.run()

    target = context_stack.data_profiling.get("target")
    if not target:
        raise ValueError("Contextualization n'a pas détecté la cible (data_profiling['target']).")

    if target not in df.columns:
        raise ValueError(f"Cible détectée '{target}' absente du dataset. Colonnes: {list(df.columns)[:20]}...")

    return context_stack, target
