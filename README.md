# Adapter - LoRA Training Server

Train LoRA adapters on any HuggingFace model, export to GGUF, and push to devices over-the-air via Supabase.

## What It Does

1. You register a base model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`)
2. You pick a dataset (Medical, Coding, Creative, General, or upload your own)
3. It trains a LoRA adapter using SFT
4. Converts both the base model and adapter to GGUF format
5. Quantizes the base model (Q2_K through Q8_0)
6. Uploads the adapter to Supabase, base model to HuggingFace
7. Android app picks it up via OTA

## Prerequisites

- **Python 3.10+**
- **llama.cpp** repo cloned somewhere (for GGUF conversion scripts)
- **Supabase** project with `base_models`, `adapters`, `adapter_deployments`, `update_logs` tables
- **GPU recommended** (CUDA) but CPU works for small models

## Setup

```bash
git clone <this-repo>
cd Adapter
./setup.sh
```

The setup script handles everything: venv, dependencies, `.env` file, and database migrations.

Or do it manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env       # then edit with your credentials
alembic upgrade head      # run database migrations
```

### Configure `.env`

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
SUPABASE_SERVICE_KEY=your-service-role-key

LLAMA_CPP_DIR=/path/to/llama.cpp

# Optional: auto-upload base model GGUF to HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO_ID=your-username/your-repo
```

### Build llama-quantize (for quantization)

```bash
cd /path/to/llama.cpp
cmake -B build
cmake --build build --target llama-quantize
```

## Run

```bash
source .venv/bin/activate
python src/gradio_trainer.py
```

Opens at **http://localhost:7860**

Options:
```bash
python src/gradio_trainer.py --port 8080        # custom port
python src/gradio_trainer.py --share             # public Gradio link
python src/gradio_trainer.py --llama-cpp-dir /x  # override llama.cpp path
```

## How to Use

### Step 1: Register a Base Model

Go to the **Base Models** tab.

| Field | What to put |
|-------|-------------|
| Model Name | Whatever you want, e.g. `Qwen-0.5B` |
| HuggingFace Model ID | The safetensors model ID, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| Version | e.g. `1.0` |
| Size (MB) | Approximate size |

The HF Model ID is what gets downloaded for training. The GGUF download link gets auto-filled after training if you have HuggingFace storage configured.

### Step 2: Train an Adapter

Go to the **Train Adapter** tab.

1. **Select base model** from the dropdown
2. **Name your adapter** (e.g. `medical-assistant`)
3. **Pick domain** (e.g. `medical`)
4. **Choose dataset source:**
   - *Predefined* -- pick from Medical, Coding, Creative, General
   - *Upload JSON* -- upload your own file
   - *Paste JSON* -- paste directly
5. **Set training params:**
   - Epochs (default 3)
   - Batch size (default 4)
   - Learning rate (default 2e-4)
   - LoRA rank (default 8) and alpha (default 32)
6. **Pick quantization** -- Q4_K_M is a good default
7. Click **Start Training**

Training streams live progress (loss, epoch, speed). When done:
- LoRA adapter GGUF is uploaded to Supabase storage
- Base model GGUF is uploaded to HuggingFace (if configured)
- Download link is auto-updated in the database

### Step 3: Manage

Go to **Manage Adapters** to view, publish, or delete adapters.

Use **Edit / Delete** sub-tab in Base Models to update model info.

## Dataset Format

JSON array of instruction/output pairs:

```json
[
  {"instruction": "Explain diabetes", "output": "Diabetes is a chronic condition..."},
  {"instruction": "What is hypertension?", "output": "Hypertension is high blood pressure..."}
]
```

Built-in datasets: **Medical** (~50), **Coding** (~60), **Creative** (~30), **General** (~40)

## Project Structure

```
Adapter/
├── setup.sh                    # Setup script
├── requirements.txt            # Python deps
├── example.env                 # Env template
├── alembic.ini                 # Migration config
├── alembic/versions/           # DB migrations
├── src/
│   ├── gradio_trainer.py       # Web UI (entry point)
│   ├── supabase_manager.py     # Supabase client
│   ├── default_datasets.py     # Built-in datasets
│   ├── models/models.py        # DB models (SQLAlchemy)
│   ├── training/gguf_trainer.py # Training + GGUF conversion
│   └── storage/                # Upload plugins (HuggingFace, extensible)
└── datasets/                   # Extra dataset files
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `LLAMA_CPP_DIR not set` | Set it in `.env` or pass `--llama-cpp-dir` |
| Out of memory during training | Use a smaller model or reduce batch size |
| Upload fails | Check Supabase credentials and that `adapters` storage bucket exists |
| `llama-quantize not found` | Build llama.cpp: `cmake -B build && cmake --build build --target llama-quantize` |
| GGUF conversion fails | Make sure `convert_lora_to_gguf.py` and `convert_hf_to_gguf.py` exist in your llama.cpp dir |
