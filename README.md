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
8. Update existing adapters with new versions — old version auto-deprecated, devices get the update

## Prerequisites

- **Python 3.10+**
- **Supabase** project (tables and RLS policies are created automatically via migrations)
- **GPU recommended** (CUDA) but CPU works for small models

## Setup

```bash
git clone <this-repo>
cd Adapter
./setup.sh
```

The setup script handles everything: venv, dependencies, `.env` file, database migrations, and RLS policies.

Or do it manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env       # then edit with your credentials
alembic upgrade head      # run database migrations + RLS policies
```

### Configure `.env`

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
SUPABASE_SERVICE_KEY=your-service-role-key

# Optional: auto-upload base model GGUF to HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO_ID=your-username/your-repo
```

### Build llama-quantize (first time only)

The GGUF conversion scripts are included in `include/llama/`. The quantize binary needs to be built once:

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build
cmake --build build --target llama-quantize
cp build/bin/llama-quantize /path/to/Adapter/include/llama/bin/
cp build/bin/lib*.so* /path/to/Adapter/include/llama/bin/
```

## Run

```bash
source .venv/bin/activate
python src/gradio_trainer.py
```

Opens at **http://localhost:7860**

Options:
```bash
python src/gradio_trainer.py --port 8080   # custom port
python src/gradio_trainer.py --share        # public Gradio link
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

### Step 3: Update an Existing Adapter

Go to **Manage Adapters** > **Update Adapter** tab.

1. **Select adapter** from the dropdown
2. Click **Load Settings** -- pre-fills name, domain, base model, and auto-bumps the version
3. **Choose dataset** and adjust training params if needed
4. **Auto-deprecate** is on by default -- old version gets deprecated and its deployment deactivated after successful training
5. Click **Retrain & Update**

The new version is deployed automatically. Android devices detect the update on next refresh.

### Step 4: Manage

Go to **Manage Adapters** > **Status / Delete** to change adapter status or delete adapters.

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

## Database & Security

Tables are created and migrated automatically via Alembic (`alembic upgrade head`):

- `base_models` -- registered HuggingFace models
- `adapters` -- trained LoRA adapters with version tracking
- `adapter_deployments` -- rollout configuration per adapter
- `update_logs` -- device-level download/install telemetry

**Row Level Security (RLS)** is applied automatically via migration:
- Android apps can read active models and published adapters (read-only)
- Only the service role (backend) can create, update, or delete records
- Devices can insert their own update logs

## Project Structure

```
Adapter/
├── setup.sh                       # Setup script
├── requirements.txt               # Python deps
├── example.env                    # Env template
├── alembic.ini                    # Migration config
├── alembic/versions/              # DB migrations + RLS policies
├── include/llama/                 # Vendored llama.cpp tools
│   ├── convert_hf_to_gguf.py     # HF -> GGUF conversion
│   ├── convert_lora_to_gguf.py   # LoRA -> GGUF conversion
│   ├── gguf-py/                   # GGUF Python package
│   └── bin/                       # llama-quantize + shared libs
├── src/
│   ├── gradio_trainer.py          # Web UI (entry point)
│   ├── supabase_manager.py        # Supabase client
│   ├── default_datasets.py        # Built-in datasets
│   ├── models/models.py           # DB models (SQLAlchemy)
│   ├── training/gguf_trainer.py   # Training + GGUF conversion
│   └── storage/                   # Upload plugins (HuggingFace, extensible)
└── datasets/                      # Extra dataset files
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of memory during training | Use a smaller model or reduce batch size |
| Upload fails | Check Supabase credentials and that `adapters` storage bucket exists |
| `llama-quantize not found` | Build it from llama.cpp source and copy to `include/llama/bin/` |
| GGUF conversion fails | Check that `include/llama/` has `convert_lora_to_gguf.py` and `convert_hf_to_gguf.py` |
| Shared lib errors when quantizing | Copy all `lib*.so*` files from llama.cpp build into `include/llama/bin/` |
| Migration fails | Check `SUPABASE_DB_URL` in `.env` and that the database is accessible |
