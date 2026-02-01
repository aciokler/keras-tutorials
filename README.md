mini GPT (Keras)
=================

This repository contains a minimal GPT-style example implemented with Keras. It includes two main scripts under `backend/minigpt/`:

- `train_gpt.py` — trains (or loads) a small GPT model and saves it as `mini_gpt.keras`.
- `inference_gpt.py` — loads `mini_gpt.keras` and provides simple text-generation helpers.

Quick checklist
- Python 3.8+ (recommended)
- GPU + CUDA drivers if you want acceleration (optional — CPU will work but be much slower)
- pip and virtualenv/venv

Install

1. From the repo root create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies from the backend requirements file:

```bash
pip install -r backend/requirements.txt
```

Run (recommended: run from `backend/`)

1. Change into the `backend` directory so model/data paths are local and simple:

```bash
cd backend
```

2. Train the model (this script will download a small dataset and a SentencePiece vocabulary automatically using `keras.utils.get_file`):

```bash
python minigpt/train_gpt.py
```

The training script will save the model as `mini_gpt.keras` in the current working directory by default.

3. Run inference (make sure `mini_gpt.keras` exists in the same directory or update `MINI_GPT_MODEL_PATH` inside the script):

```bash
python minigpt/inference_gpt.py
```

Environment / tips
- The scripts set `XLA_PYTHON_CLIENT_MEM_FRACTION=1.00` internally to affect GPU memory usage. You can override or set `CUDA_VISIBLE_DEVICES` before running to choose GPUs, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 python minigpt/train_gpt.py
```

- Keras will cache downloaded files (dataset and vocabulary) in the Keras cache (commonly `~/.keras/datasets/`).

Troubleshooting
- OOM on GPU: lower `batch_size` or `sequence_length` inside the scripts, or run on CPU.
- Missing packages / installation failures: re-run `pip install -r backend/requirements.txt` and ensure a recent pip version.
- If the scripts can't find `mini_gpt.keras`, ensure you run from `backend/` or set `MINI_GPT_MODEL_PATH` to an absolute path.

Next steps / suggestions
- Add small shell wrappers (e.g. `run_train.sh`, `run_infer.sh`) or a Makefile to standardize commands.
- Parameterize training hyperparameters and the model path with CLI flags or environment variables.
