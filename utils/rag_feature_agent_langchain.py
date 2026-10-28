#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Feature Agent (with Arguments-based task mapping)
-----------------------------------------------------
- Reads task list from arguments.global_conxtexts
- Asks LLM to pick the most relevant one
- Returns both task name and ID
"""

import os, json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_selection import (
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from openai import OpenAI
from utils.Arguments import arguments


from langchain_community.tools.tavily_search import TavilySearchResults


os.environ["OPENAI_API_KEY"] = arguments.api_key


# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------
def _openai_client() -> OpenAI:
    return OpenAI()  # uses OPENAI_API_KEY from arguments.py




def web_research(prompt: str, max_results: int = 3) -> List[str]:
    """
    Perform a quick web search to gather contextual information
    related to the user's analytical objective.
    """
    try:
        tavily = TavilySearchResults(max_results=max_results)
        results = tavily.run(prompt)
        # Extract relevant snippets or text summaries
        if isinstance(results, list):
            return [r.get("snippet", "") for r in results if isinstance(r, dict)]
        elif isinstance(results, str):
            return [results]
        else:
            return []
    except Exception as e:
        print(f"[WARN] Web research failed: {e}")
        return []


def detect_task_via_llm(prompt: str, columns: List[str], model: Optional[str] = None) -> str:
    """
    Asks the LLM to choose the most relevant task name from arguments.global_conxtexts.
    """
    model = model or arguments.model
    client = _openai_client()

    possible_tasks = list(arguments.global_conxtexts.keys())
    task_list = "\n".join(f"- {t}" for t in possible_tasks)

    sys_msg = (
        "You are an AI assistant specialized in data science use-cases. "
        "You will receive:\n"
        "1. A user's objective.\n"
        "2. The list of dataset columns.\n"
        "3. A predefined list of possible tasks.\n\n"
        "Choose ONLY one of the predefined tasks that best fits the user's intent "
        "and the dataset nature.\n\n"
        "Return JSON only in the format:\n"
        "{\"task\": \"<exact task name from the list>\"}"
    )

    user_msg = (
        f"User request:\n{prompt}\n\n"
        f"Dataset columns:\n" + "\n".join(f"- {c}" for c in columns) + "\n\n"
        f"Available tasks:\n{task_list}\n\n"
        "Your output must be one of these task names."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}]
    )
    content = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(content)
        task = str(data.get("task", "")).strip()
        if task not in arguments.global_conxtexts:
            raise ValueError
        return task
    except Exception:
        # fallback heuristic
        text = prompt.lower()
        if "forecast" in text or "time" in text:
            return "Time Series Forecasting"
        elif "class" in text or "label" in text:
            return "Classification"
        elif "cluster" in text:
            return "Clustering"
        elif "anomaly" in text:
            return "Anomaly Detection"
        else:
            return "Regression"


def detect_target_via_llm(columns: List[str], prompt: str, model: Optional[str] = None) -> str:
    """LLM chooses the most relevant target column from dataset."""
    model = model or arguments.model
    client = _openai_client()

    sys_msg = (
        "You are a data assistant. Given dataset columns and a user's objective, "
        "choose ONE column to be the prediction target. "
        "Return JSON: {\"target\":\"column_name\"}."
    )
    user_msg = f"User objective:\n{prompt}\n\nColumns:\n" + "\n".join(f"- {c}" for c in columns)

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}]
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        target = data.get("target", "")
    except Exception:
        target = ""
    return target


def detect_usecase_via_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Ask LLM to pick the most relevant use case from arguments.usecases based on the prompt.
    """
    model = model or arguments.model
    client = _openai_client()

    possible_ucs = list(arguments.usecases.keys())
    uc_list = "\n".join(f"- {uc}: {arguments.usecases[uc]['title']}" for uc in possible_ucs)

    sys_msg = (
        "You are an assistant that identifies the most relevant use case "
        "for a given user request in a sports analytics or ML context. "
        "Return JSON {\"usecase\": \"<exact use case ID>\"}."
    )

    user_msg = f"User request:\n{prompt}\n\nAvailable use cases:\n{uc_list}"

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}]
    )

    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        uc = data.get("usecase", "")
        if uc not in arguments.usecases:
            raise ValueError
        return uc
    except Exception:
        # fallback heuristic
        text = prompt.lower()
        if "blessure" in text or "injury" in text:
            return "UC-IP"
        elif "effort" in text or "rpe" in text:
            return "UC-RPE"
        elif "match" in text or "victoire" in text or "défaite" in text:
            return "UC-PRM"
        elif "recrut" in text or "joueur" in text:
            return "UC-REC"
        else:
            return "UC-RPE"




# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _normalize(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by handling NaN values safely.
    - Numeric columns: fill with mean.
    - Categorical columns: fill with mode.
    """
    df = df.copy()
    for c in df.columns:
        if _is_num(df[c]):
            df[c] = df[c].fillna(df[c].mean())
        else:
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "missing")
    return df


def build_doc(name: str, s: pd.Series) -> str:
    """Create a textual description of a column."""
    return f"column:{name} dtype:{s.dtype} unique:{s.nunique(dropna=True)}"



def rag_scores(docs: List[str], query: str) -> np.ndarray:
    vec = TfidfVectorizer(min_df=1, max_features=4096)
    X = vec.fit_transform(docs + [query])
    return _normalize(linear_kernel(X[-1], X[:-1]).ravel())




def stat_scores(X: pd.DataFrame, y: pd.Series, task: str) -> np.ndarray:
    num_mask = np.array([_is_num(X[c]) for c in X.columns])
    X_num = X.loc[:, num_mask].copy()
    y_use = pd.factorize(y)[0] if task == "Classification" else pd.to_numeric(y, errors="coerce")

    scores = np.zeros(len(X.columns))
    if not X_num.empty:
        Xn = X_num.to_numpy(dtype=float, copy=False)
        if task == "Classification":
            fvals, _ = f_classif(Xn, y_use)
            mi = mutual_info_classif(Xn, y_use, random_state=0)
        else:
            fvals, _ = f_regression(Xn, y_use)
            mi = mutual_info_regression(Xn, y_use, random_state=0)
        scores[num_mask] = 0.5 * _normalize(fvals) + 0.5 * _normalize(mi)
    return _normalize(scores)


# ---------------------------------------------------------------------
# Main Agent Logic
# ---------------------------------------------------------------------

def run_agent(df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
    columns = list(df.columns)
    task_name = detect_task_via_llm(prompt, columns)
    target = detect_target_via_llm(columns, prompt)
    if target not in df.columns:
        target = columns[-1]
    use_case = detect_usecase_via_llm(prompt)

    # Retire la target des features candidates
    features = [c for c in columns if c != target]

    client = _openai_client()
    model = arguments.model

    sys_msg = (
        "You are an expert data scientist. "
        "Given a user's objective and dataset columns, "
        "select the most relevant columns (based only on their names). "
        "Return valid JSON ONLY in this format:\n"
        "{\"relevant_features\": [\"col1\", \"col2\", ...]}"
    )

    user_msg = (
        f"User objective:\n{prompt}\n\n"
        f"Columns:\n" + "\n".join(f"- {c}" for c in features) + "\n\n"
        "Select only the relevant ones (5–10 max)."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    content = (resp.choices[0].message.content or "").strip()
    relevant_features = []

    # --- Étape 1 : tentative de JSON propre ---
    try:
        # Nettoyage des éventuelles lignes parasites
        json_str = content.split("```")[-1].strip() if "```" in content else content
        json_str = json_str.replace("json", "").strip()
        data = json.loads(json_str)
        if isinstance(data, dict) and "relevant_features" in data:
            relevant_features = [
                f for f in data["relevant_features"] if isinstance(f, str) and f in features
            ]
    except Exception as e:
        print(f"[WARN] JSON parsing failed: {e}\n→ Raw content: {content}")

    # --- Étape 2 : fallback sémantique ---
    if not relevant_features:
        relevant_features = [
            c for c in features
            if any(kw in c.lower() for kw in ["score", "value", "rate", "time", "date"])
        ]

    # --- Étape 3 : fallback par défaut ---
    if not relevant_features and features:
        relevant_features = features[:5]



    """
    X = df.drop(columns=[target])
    y = df[target]

    # >>> Handle NaNs safely
    X = _clean_dataframe(X)
    if _is_num(y):
        y = y.fillna(y.mean())
    else:
        mode = y.mode(dropna=True)
        y = y.fillna(mode.iloc[0] if not mode.empty else "missing")

    # Continue with normal flow
    docs = [build_doc(c, X[c]) for c in X.columns]
    rag_vec = rag_scores(docs, prompt)
    stat_vec = stat_scores(X, y, task_name)

    final = 0.6 * stat_vec + 0.4 * rag_vec
    table = pd.DataFrame({
        "feature": X.columns,
        "final_score": final
    }).sort_values("final_score", ascending=False)"""
    
    return {
        "context_type": task_name,
        "context_id": arguments.global_conxtexts.get(task_name, -1),
        "target_column": target,
        "use_case": use_case,
        "use_case_title": arguments.usecases.get(use_case, {}).get("title", ""),
        "relevant_features": relevant_features or []
    }
# Wrapper Class
# ---------------------------------------------------------------------
class RAGFeatureAgent:
    """Used inside Contextualization pipeline."""

    def __init__(self):
        pass
    def analyze(self, df: pd.DataFrame, user_prompt: str) -> Dict[str, Any]:
        return run_agent(df, user_prompt)


    