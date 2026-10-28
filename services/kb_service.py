# services/kb_service.py
import os
import time
import subprocess
from pathlib import Path
import requests

from utils.Arguments import arguments as args
from modules.KnowledgeBase import KnowledgeBase

def run_weaviate_bash():
    script_path = os.path.join(os.getcwd(), "modules/run_weaviate.sh")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    subprocess.run(["/bin/bash", script_path], check=True)

def wait_for_weaviate_ready(url="http://localhost:8080/v1/.well-known/ready", timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Weaviate not ready within timeout.")

def build_kb(drop_existing: bool = False) -> KnowledgeBase:
    run_weaviate_bash()
    wait_for_weaviate_ready()

    kb = KnowledgeBase(
        weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        openai_api_key=args.api_key,
        create_if_missing=True,
        drop_existing=drop_existing,
    )

    records = kb.load_json_folder("datasets/json_img_dataset2/*")
    if records:
        kb.batch_insert(records, batch_size=64)
    return kb
