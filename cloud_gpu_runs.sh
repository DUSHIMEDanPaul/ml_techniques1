#!/usr/bin/env bash
# Remaining Transformer experiments - run on a GPU machine (e.g. Google Colab T4, Kaggle P100).
# Total on a T4: roughly 30-45 minutes.
#
# Colab:
#   !unzip -q cloud_bundle.zip -d ml_techniques1      (or: !git clone <your repo> ml_techniques1)
#   %cd ml_techniques1
#   !bash cloud_gpu_runs.sh
#   then download results.zip and final_model.zip (see the end of this script)
set -e
export PYTHONUTF8=1 TRANSFORMERS_VERBOSITY=error
cd "$(dirname "$0")"
pip uninstall -y -q torchao 2>/dev/null || true   # old Colab torchao breaks peft LoRA
pip install -q transformers peft scikit-learn pandas scipy gensim emoji sentencepiece protobuf matplotlib tabulate joblib
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found - enable a GPU runtime'; print(torch.cuda.get_device_name(0))"
cd src

# 1. AfroXLMR-small, full fine-tuning (ran out of RAM on the 8 GB laptop)
python -u finetune_transformer.py --model Davlan/afro-xlmr-small --name afroxlmr_small_ft --lr 3e-5 --epochs 5 --save 2>&1 | tee ../results/log_afroxlmr_small_ft.txt
# 2. AfroXLMR-small, LoRA (parameter-efficient: ~1.2M trainable params)
python -u finetune_transformer.py --model Davlan/afro-xlmr-small --name afroxlmr_small_lora --lora --lora_r 16 --lr 5e-4 --epochs 5 --save 2>&1 | tee ../results/log_afroxlmr_small_lora.txt
# 3. AfroXLMR-base (270M params), full fine-tuning - the largest model we can afford
python -u finetune_transformer.py --model Davlan/afro-xlmr-base --name afroxlmr_base_ft --lr 2e-5 --epochs 5 --save 2>&1 | tee ../results/log_afroxlmr_base_ft.txt
# 4. KinyaBERT-small (monolingual Kinyarwanda BERT), full fine-tuning
python -u finetune_transformer.py --model jean-paul/KinyaBERT-small --name kinyabert_small_ft --lr 5e-5 --epochs 6 --save 2>&1 | tee ../results/log_kinyabert_small_ft.txt
# 5. AfriBERTa-small with a frozen encoder (linear probe) - how much does fine-tuning the encoder matter?
python -u finetune_transformer.py --model castorini/afriberta_small --name afriberta_small_frozen --freeze --lr 1e-3 --epochs 6 2>&1 | tee ../results/log_afriberta_small_frozen.txt

# Pick the best model on the DEV set -> models/final, then evaluate it
FINAL=$(python select_final.py | tail -1)
python analyze.py --best "$FINAL" --baseline tfidf_char_lr --compare bilstm_fasttext_attn afriberta_small_ft
python probes.py --model ../models/final

cd ..
zip -qr results.zip results
zip -qr final_model.zip models/final
echo "DONE. Download results.zip and final_model.zip and unzip them into the project folder (overwrite)."
