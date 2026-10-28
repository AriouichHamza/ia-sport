
import os
import time
import subprocess
from pathlib import Path

import pandas as pd
import requests

from utils.Arguments import arguments as args
from modules.DataLoader import DataLoader
from modules.Contextualization import Contextualization
from modules.ContextSctack import ContextSctack
from modules.KnowledgeBase import KnowledgeBase

from models.classification_models import run_classification_model
from models.regression_models import run_regression_model


user_prompt1 = "Prédire la probabilité de blessure des joueurs pendant l'entraînement"
user_prompt2 = "prediire le RPE en fonction des données d'entrainement"

def run_weaviate_bash():
    """Runs the Weaviate launch script only if the container isn't running."""
    script_path = os.path.join(os.getcwd(), "modules/run_weaviate.sh")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"❌ Script not found at: {script_path}")
    print("[INFO] Checking Weaviate container via bash script...")
    try:
        result = subprocess.run(
            ["/bin/bash", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to execute run_weaviate.sh:\n{e.stderr}")


def wait_for_weaviate_ready(url="http://localhost:8080/v1/.well-known/ready", timeout=60):
    """Wait until the Weaviate REST API is reachable and ready."""
    print("[INFO] Waiting for Weaviate to become ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("[READY] Weaviate is live and ready to accept connections.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Weaviate did not become ready within the timeout period.")


def build_kb():
    run_weaviate_bash()
    wait_for_weaviate_ready()

    kb = KnowledgeBase(
        weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        openai_api_key=args.api_key,
        create_if_missing=True,
        drop_existing=False,  # IMPORTANT: do not drop each loop
    )

    # Ingest JSON folder
    records = kb.load_json_folder("datasets/json_img_dataset2/*")
    print(Path("datasets/json_img_dataset2/").absolute())
    if records:
        kb.batch_insert(records, batch_size=64)

    print("Total objets:", kb.count())
    return kb


kb = None


def main1():
    # --------------------------------------------------
    # 1) Explicit injury dataset (NO loop)
    # --------------------------------------------------
    dataset_path = Path("datasets/day_approach_maskedID_timeseries.csv")

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    print("\n" + "=" * 100)
    print("[DATASET]", dataset_path)
    print("=" * 100)

    # --------------------------------------------------
    # 2) Load dataset
    # --------------------------------------------------
    data_loader = DataLoader(dataset_path)
    df = data_loader.load_dataset()
    print("[INFO] Loaded shape:", df.shape)

    # --------------------------------------------------
    # 3) Contextualization (still useful even in no_kb)
    # --------------------------------------------------
    context_stack = ContextSctack()
    contextualizer = Contextualization(
        dataframe=df,
        user_prompt=user_prompt1,
        context_stack=context_stack
    )
    contextualizer.run()

    print("[✔] Global context:", context_stack.global_context)
    print("[✔] Use case:", context_stack.use_case)
    print("[✔] Target:", context_stack.data_profiling.get("target"))
    print("[✔] Features:", context_stack.data_profiling.get("features", [])[:20], "...")

    target_column = context_stack.data_profiling.get("target")
    if not target_column or target_column not in df.columns:
        print(f"[ERROR] Invalid target detected: {target_column}")
        return

    print("[DTYPES]\n", df.dtypes)

    # --------------------------------------------------
    # 4) Run classification — PURE LOCAL MODE
    # --------------------------------------------------
    build_kb()
    run_classification_model(
        df=df,
        target_column=target_column,
        model_name=None,          # try all models
        kb=None,                  # optional, ignored in no_kb
        context_stack=context_stack,
        kb_mode="kb_low",         
        random_pipelines=False,
        num_random_pipelines=3,
        random_state=42,
    )




def main2():
    # ------------------------------------------------------------------
    # Dataset path (fixe ou via args si tu veux)
    # ------------------------------------------------------------------
    dataset_path = Path("datasets/iot_sports_training_dataset.csv")
    data_loader = DataLoader(dataset_path)
    df = data_loader.load_dataset()
    print("[INFO] Loaded shape:", df.shape)
    print("[✔] DataFrame received:", df.shape[0], "rows ×", df.shape[1], "columns")

    # Contextualisation (la bonne classe)
    
    context_stack = ContextSctack()
    contextualisation_recom = Contextualization(
        dataframe=df,
        user_prompt=user_prompt2,
        context_stack=context_stack
    )
    contextualisation_recom.run()

    print("[✔] Global context:", context_stack.global_context)
    print("[✔] Use case:", context_stack.use_case)
    print("[✔] Target:", context_stack.data_profiling.get("target"))
    print("[✔] Features:", context_stack.data_profiling.get("features", [])[:6], "...")

    target_column = context_stack.data_profiling.get("target")
    if not target_column:
        raise ValueError("Contextualization n'a pas détecté la cible (data_profiling['target']).")

    build_kb()  
    
    run_regression_model(
        df=df,
        target_column=target_column,
        model_name=None,
        kb=kb,  
        context_stack=context_stack,
        kb_mode="kb_low",
        random_pipelines=True,
    )



if __name__ == "__main__":
    main2()
    main1()
