"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SEED = 42

# AfriSenti-SemEval 2023, Kinyarwanda (Muhammad et al., 2023)
AFRISENTI_BASE = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main"
SPLIT_URLS = {
    "train": f"{AFRISENTI_BASE}/data/kin/train.tsv",
    "dev": f"{AFRISENTI_BASE}/data/kin/dev.tsv",
    "test": f"{AFRISENTI_BASE}/data/kin/test.tsv",
}
ANNOTATOR_URL = f"{AFRISENTI_BASE}/data_with_annotators_labels/kinyarwanda_individual_labels.csv"

for d in (PROC_DIR, RESULTS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
