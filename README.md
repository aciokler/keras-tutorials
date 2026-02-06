Backend folder - quick reference
=============================

What’s in `backend/` (short)
- `requirements.txt` — Python dependencies to install with pip
- `mini_gpt.keras` — saved model file (created after training)
- `minigpt/` package:
  - `train_gpt.py` — trains (or loads) the model; downloads dataset & vocabulary automatically
  - `inference_gpt.py` — loads the model and runs text generation
- `testcuda.py` — helper to check CUDA/GPU availability

Quick commands (run from repo root or from `backend/`)

```bash
# from repo root:
cd backend
pip install -r requirements.txt
python minigpt/train_gpt.py      # train or save model as mini_gpt.keras
python minigpt/inference_gpt.py  # run inference (requires mini_gpt.keras)
```

Notes
- Model path: `mini_gpt.keras` is saved/loaded relative to the current working directory.
- To select GPUs: set `CUDA_VISIBLE_DEVICES` before running, e.g. `CUDA_VISIBLE_DEVICES=0 python minigpt/train_gpt.py`.
- Keras caches downloaded files (dataset & vocabulary), typically under `~/.keras/`.

That's it — short and actionable. Want me to add simple shell wrappers (`backend/run_train.sh`, `backend/run_infer.sh`)?
