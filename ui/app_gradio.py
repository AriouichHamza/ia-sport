import os
import time
import subprocess
import pandas as pd
from pathlib import Path
import gradio as gr
import warnings
import logging

# Suppress resource and library-specific warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*no meaningful features.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
logging.getLogger("lightgbm").setLevel(logging.ERROR) 

# Load config
from ui.ui_config import DEFAULT_SERVER_NAME, DEFAULT_PORT, APP_TITLE, APP_DESCRIPTION
from utils.Arguments import arguments as args

# --- Custom CSS ---
custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #f8f9fa; font-family: 'Inter', sans-serif;}
#header-text {text-align: center; margin-bottom: 20px;}
#run-btn {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    color: white;
    font-weight: bold;
    border: none;
    height: 50px;
}
#run-btn:hover {
    transform: scale(1.02);
    transition: 0.2s;
}
.winner-card {
    padding: 20px;
    background: white;
    border-radius: 12px;
    border-left: 8px solid #FFD700;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
"""

# --- Infrastructure & KB Setup (Kept as is) ---
def ensure_weaviate_is_ready():
    base_dir = Path(__file__).resolve().parent.parent
    script_path = base_dir / "modules" / "run_weaviate.sh"
    try:
        subprocess.run(["/bin/bash", str(script_path)], check=True, timeout=30)
        import requests
        for _ in range(15):
            try:
                if requests.get("http://127.0.0.1:8080/v1/.well-known/ready", timeout=2).status_code == 200:
                    return True
            except:
                time.sleep(2)
    except Exception as e:
        print(f"⚠️ Infrastructure bypassed: {e}")
    return False

ensure_weaviate_is_ready()

from modules.KnowledgeBase import KnowledgeBase
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080")
global_kb = None
try:
    global_kb = KnowledgeBase(weaviate_url=WEAVIATE_URL, openai_api_key=args.api_key, create_if_missing=True)
    base_dir = Path(__file__).resolve().parent.parent
    json_pattern = str(base_dir / "datasets" / "json_img_dataset2" / "*")
    records = global_kb.load_json_folder(json_pattern)
    if records and global_kb.count() == 0:
        global_kb.batch_insert(records)
except Exception as e:
    print(f"⚠️ KB Initialization failed: {e}")

kb = global_kb

# --- UI LOGIC ---

def ui_run(dataset_file, user_prompt, force_task, kb_mode, random_pipelines, num_random_pipelines):
    from core.pipeline_runner import run_job
    import pandas as pd
    
    # 1. Determine Path
    path_to_use = dataset_file if isinstance(dataset_file, str) else dataset_file.name
    
    # 2. Execute Backend
    # summary: The string containing "Best: MLPClassifier | pipeline=..."
    # table: The DataFrame (currently returning empty)
    # logs: The captured stdout
    summary, table, logs = run_job(
        dataset_path=path_to_use,
        user_prompt=user_prompt,
        force_task=force_task,
        kb=kb,
        kb_mode=kb_mode,
        random_pipelines=random_pipelines,
        num_random_pipelines=int(num_random_pipelines),
        random_state=42,
    )
    
    # 3. Handle Visual Summary
    # If the backend built a summary string, we display it nicely
    if summary and "Best:" in summary:
        # Extract the 'Best' line for a clean look
        best_info = summary.split("Best:")[-1].strip()
        
        summary_md = f"""
<div class="winner-card" style="padding: 20px; background: white; border-radius: 12px; border-left: 8px solid #FFD700; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <h2 style="margin-top:0; color: #182848;">🏆 Analysis Complete</h2>
    <p style='font-size: 1.1em;'>The pipeline identified the following optimal configuration:</p>
    <div style="background: #f1f3f5; padding: 15px; border-radius: 8px; font-family: monospace;">
        {best_info}
    </div>
    <p style="margin-top: 10px; font-size: 0.9em; color: #666;">Full details are available in the <strong>Detailed Logs</strong> tab.</p>
</div>
"""
    else:
        summary_md = "### ⚠️ Pipeline executed but no 'Best' model was identified in the summary."

    # 4. Handle the Table
    # If table is empty, we don't want to show an empty box. 
    # We can try to show at least the top results if the DataFrame works, 
    # otherwise we keep it as is.
    display_table = table if not table.empty else pd.DataFrame([{"Status": "Check Logs Tab for tabular data"}])

    return summary_md, display_table, logs



def main():
    with gr.Blocks(title=APP_TITLE, css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        
        with gr.Column(elem_id="header-text"):
            gr.Markdown(f"# 🚀 {APP_TITLE}")
            gr.Markdown(f"*{APP_DESCRIPTION}*")

        with gr.Row():
            # --- Left Column: Inputs ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📁 Data Source")
                    dataset_input = gr.File(
                        label="Upload CSV Dataset",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    dataset_path_display = gr.Textbox(
                        label="Active Dataset Path",
                        value="datasets/iot_sports_training_dataset.csv",
                        interactive=False
                    )
                    user_prompt = gr.Textbox(
                        label="Objective / Target Prediction",
                        value="predire le RPE en fonction des données d'entrainement",
                        lines=3
                    )
                
                with gr.Accordion("⚙️ Advanced Pipeline Settings", open=False):
                    with gr.Row():
                        force_task = gr.Dropdown(choices=["auto", "classification", "regression"], value="auto", label="Task Type")
                        kb_mode = gr.Dropdown(choices=["no_kb", "kb_low", "kb_high"], value="no_kb", label="KB Mode")
                    num_random_pipelines = gr.Number(value=3, precision=0, label="Search Iterations")
                    random_pipelines = gr.Checkbox(value=True, label="Enable Random Pipeline Search")

                run_btn = gr.Button("🚀 START PIPELINE", elem_id="run-btn", variant="primary")

            # --- Right Column: Results ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📊 Summary & Metrics"):
                        out_summary = gr.HTML("<div style='text-align: center; padding: 50px; color: grey;'>Awaiting model execution...</div>")
                        gr.Markdown("### 🥈 Top 3 Performance Leaderboard")
                        out_table = gr.Dataframe(
                            label="Best Models",
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.TabItem("📜 Detailed Logs"):
                        out_logs = gr.Code(
                            label="System Logs",
                            language="shell",
                            lines=25,
                            interactive=False
                        )

        # Sync path display
        dataset_input.change(
            fn=lambda x: x.name if x else "datasets/iot_sports_training_dataset.csv",
            inputs=[dataset_input],
            outputs=[dataset_path_display]
        )

        # Run logic
        run_btn.click(
            fn=ui_run,
            inputs=[dataset_input, user_prompt, force_task, kb_mode, random_pipelines, num_random_pipelines],
            outputs=[out_summary, out_table, out_logs],
        )

    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=int(os.getenv("PORT", str(DEFAULT_PORT))),
        show_error=True,
    )

if __name__ == "__main__":
    main()