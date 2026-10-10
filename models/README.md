# models/

Large artefacts are not stored in git (see `.gitignore`).

| file / folder | produced by | notes |
|---|---|---|
| `tfidf_baseline.joblib` | `src/baselines.py` | TF-IDF char n-gram + Logistic Regression (committed, 1.3 MB, used by the app as a comparison) |
| `w2v_kin.kv*`, `fasttext_kin.kv*` | `src/train_embeddings.py` | Word2Vec / FastText embeddings (~0.9 GB, re-create locally) |
| `<experiment>/` | `src/finetune_transformer.py --save` | fine-tuned Transformers |
| `final/` | `src/select_final.py` | best model on dev (AfroXLMR-small, full fine-tuning). Published on the Hugging Face Hub, see the main README |
