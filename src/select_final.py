"""Pick the Transformer with the best *dev* macro-F1 among the saved models and copy it to models/final.

Usage:  python src/select_final.py
"""
import json
import shutil

from config import MODELS_DIR, RESULTS_DIR

rows = [json.loads(l) for l in (RESULTS_DIR / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
saved = [r for r in rows if r["family"] == "transformer" and (MODELS_DIR / r["name"]).exists()]
best = max(saved, key=lambda r: r["dev"]["macro_f1"])
dst = MODELS_DIR / "final"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(MODELS_DIR / best["name"], dst)
(dst / "selected_experiment.txt").write_text(best["name"])
print(f"final model = {best['name']}  (dev macro-F1 {best['dev']['macro_f1']}, test macro-F1 {best['test']['macro_f1']})")
print(best["name"])
