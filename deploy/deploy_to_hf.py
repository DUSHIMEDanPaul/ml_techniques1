"""Deploy the final model + Gradio app to Hugging Face (free CPU Space).

What it does
  1. uploads models/final            -> model repo  <user>/kinyarwanda-sentiment-afroxlmr
  2. creates a Gradio Space           -> space repo  <user>/kinyarwanda-sentiment
     with app.py, src/preprocess.py, the TF-IDF baseline and deploy/space_requirements.txt
  3. sets the Space variable MODEL_ID so app.py loads the model from the Hub

Prerequisites
  pip install huggingface_hub
  huggingface-cli login        (token with *write* access, from https://huggingface.co/settings/tokens)

Usage
  python deploy/deploy_to_hf.py --user <your-hf-username>
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Kinyarwanda Sentiment
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
---

Kinyarwanda tweet sentiment classifier (negative / neutral / positive): AfroXLMR-small fine-tuned
on AfriSenti-SemEval 2023 Kinyarwanda. Source code: see the GitHub repository linked in the project README.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model_repo", default="kinyarwanda-sentiment-afroxlmr")
    ap.add_argument("--space", default="kinyarwanda-sentiment")
    ap.add_argument("--skip_model", action="store_true",
                    help="model already pushed (e.g. with deploy/push_model.py); only create the Space")
    a = ap.parse_args()
    api = HfApi()
    model_id, space_id = f"{a.user}/{a.model_repo}", f"{a.user}/{a.space}"

    if not a.skip_model:
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(ROOT / "models" / "final"), repo_id=model_id, repo_type="model")
        print(f"model uploaded: https://huggingface.co/{model_id}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(ROOT / "app.py", tmp / "app.py")
        (tmp / "src").mkdir()
        shutil.copy(ROOT / "src" / "preprocess.py", tmp / "src" / "preprocess.py")
        (tmp / "models").mkdir()
        shutil.copy(ROOT / "models" / "tfidf_baseline.joblib", tmp / "models" / "tfidf_baseline.joblib")
        shutil.copy(ROOT / "deploy" / "space_requirements.txt", tmp / "requirements.txt")
        (tmp / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.add_space_variable(space_id, "MODEL_ID", model_id)
        api.upload_folder(folder_path=str(tmp), repo_id=space_id, repo_type="space")
    print(f"space deployed: https://huggingface.co/spaces/{space_id}  (first build takes a few minutes)")


if __name__ == "__main__":
    main()
