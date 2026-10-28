# core/pipeline_runner.py
from typing import Optional, Tuple
import pandas as pd

from core.dataset_service import load_dataset
from core.context_service import build_context
from core.result_schema import JobResult
from services.logging_service import capture_logs

from models.classification_models import run_classification_model
from models.regression_models import run_regression_model


def _run_pipeline_internal(
    dataset_path: str,
    user_prompt: str,
    force_task: str,                 # "auto" | "classification" | "regression"
    kb=None,
    kb_mode: str = "no_kb",
    random_pipelines: bool = True,
    num_random_pipelines: int = 3,
    random_state: int = 42,
) -> JobResult:
    df = load_dataset(dataset_path)

    # contextualisation -> task + target
    context_stack, target = build_context(df, user_prompt)

    # Déterminer le type de tâche
    task = context_stack.global_context  # attendu "Classification" ou "Regression"
    if isinstance(task, str):
        task_lower = task.lower()
    else:
        task_lower = "auto"

    if force_task != "auto":
        task_lower = force_task

    # Exécuter
    if "regress" in task_lower:
        # run_regression_model doit idéalement retourner un df résultats
        results = run_regression_model(
            df=df,
            target_column=target,
            model_name=None,
            kb=kb,
            context_stack=context_stack,
            kb_mode=kb_mode,
            random_pipelines=random_pipelines,
        )
        table = results if isinstance(results, pd.DataFrame) else pd.DataFrame()
        summary = f"Regression | target={target} | use_case={context_stack.use_case}"
        return JobResult(task_type="regression", summary=summary, table=table, logs="")

    # sinon classification
    results = run_classification_model(
        df=df,
        target_column=target,
        model_name=None,
        kb=kb,
        context_stack=context_stack,
        kb_mode=kb_mode,
        random_pipelines=random_pipelines,
        num_random_pipelines=num_random_pipelines,
        random_state=random_state,
    )
    table = results if isinstance(results, pd.DataFrame) else pd.DataFrame()
    summary = f"Classification | target={target} | use_case={context_stack.use_case}"
    return JobResult(task_type="classification", summary=summary, table=table, logs="")


def run_job(
    dataset_path: str,
    user_prompt: str,
    force_task: str = "auto",
    kb=None,
    kb_mode: str = "no_kb",
    random_pipelines: bool = True,
    num_random_pipelines: int = 3,
    random_state: int = 42,
) -> Tuple[str, pd.DataFrame, str]:
    """
    Fonction appelée par Gradio: renvoie (summary, table_df, logs).
    """
    def _fn():
        return _run_pipeline_internal(
            dataset_path=dataset_path,
            user_prompt=user_prompt,
            force_task=force_task,
            kb=kb,
            kb_mode=kb_mode,
            random_pipelines=random_pipelines,
            num_random_pipelines=num_random_pipelines,
            random_state=random_state,
        )

    result_obj, logs = capture_logs(_fn)

    # si crash
    if result_obj is None:
        return "Erreur: aucun résultat", pd.DataFrame(), logs

    # Injecter logs
    result_obj.logs = logs

    # Si aucun tableau renvoyé par tes modèles, on renvoie vide mais on garde logs
    table = result_obj.table if isinstance(result_obj.table, pd.DataFrame) else pd.DataFrame()

    # Résumé un peu plus utile si table présente
    if not table.empty and "recall_class_1" in table.columns and "accuracy" in table.columns:
        best = table.sort_values(["recall_class_1", "accuracy"], ascending=False).iloc[0]
        result_obj.summary += (
            f"\nBest: {best.get('model')} | pipeline={best.get('pipeline')} "
            f"| recall1={best.get('recall_class_1'):.3f} | acc={best.get('accuracy'):.3f}"
        )

    return result_obj.summary, table, result_obj.logs
