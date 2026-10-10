#!/usr/bin/env bash
# Reproduce the full pipeline. Steps 1-4 run fine on a CPU laptop (~15 min);
# the Transformer runs (step 5) need a GPU in practice (see cloud_gpu_runs.sh, ~30-45 min on a T4).
set -e
export PYTHONUTF8=1
cd "$(dirname "$0")/src"
python prepare_data.py                     # 1. download + clean + EDA
python baselines.py                        # 2. majority / NB / LR / SVM
python train_embeddings.py                 # 3. Word2Vec + FastText on KINNEWS + train tweets
python train_rnn.py                        # 4. BiLSTM / BiGRU experiments (3 seeds each)
# 5. Transformer experiments
python finetune_transformer.py --model castorini/afriberta_small --name afriberta_small_ft --lr 5e-5 --epochs 6 --save
python finetune_transformer.py --model castorini/afriberta_small --name afriberta_small_frozen --freeze --lr 1e-3 --epochs 6
python finetune_transformer.py --model jean-paul/KinyaBERT-small --name kinyabert_small_ft --lr 5e-5 --epochs 6 --save
python finetune_transformer.py --model Davlan/afro-xlmr-small --name afroxlmr_small_ft --lr 3e-5 --epochs 5 --save
python finetune_transformer.py --model Davlan/afro-xlmr-small --name afroxlmr_small_lora --lora --lora_r 16 --lr 5e-4 --epochs 4
python finetune_transformer.py --model Davlan/afro-xlmr-base --name afroxlmr_base_ft --lr 2e-5 --epochs 5 --save
# 6. model selection on dev, evaluation, error analysis, probes
FINAL=$(python select_final.py | tail -1)
python analyze.py --best "$FINAL" --baseline tfidf_char_lr --compare bilstm_fasttext_attn afriberta_small_ft
python probes.py --model ../models/final
