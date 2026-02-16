import argparse
import json
import os
import sys

# Ensure project root is on sys.path so `python src/gradio_trainer.py` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import pandas as pd

from src import default_datasets
from src.models.models import BaseModel, Adapter
from src.supabase_manager import SupabaseManager
from src.training import GGUFAdapterTrainer, is_gguf_available, QUANT_TYPES
from src.storage import get_storage_provider, get_available_providers

sm = SupabaseManager()
gguf_trainer_instance = None


def get_base_models():
    session = sm.get_db_session()
    try:
        models = session.query(BaseModel).all()
        return [(f"{m.name} ({m.version})", m.id) for m in models]
    finally:
        session.close()


def get_adapters_dataframe():
    session = sm.get_db_session()
    try:
        adapters = session.query(Adapter).all()
        data = [{
            'ID': a.id,
            'Name': a.name,
            'Domain': a.domain,
            'Version': a.version,
            'Size (MB)': a.size_mb,
            'Status': a.status,
            'Published': a.is_published,
            'Created': a.created_at.strftime('%Y-%m-%d %H:%M')
        } for a in adapters]
        return pd.DataFrame(data)
    finally:
        session.close()


def get_predefined_datasets():
    return {
        "Medical": default_datasets.medical_data,
        "Coding": default_datasets.coding_data,
        "Creative": default_datasets.creative_data,
        "General": default_datasets.general_data,
    }


def add_base_model(name, hf_model_id, download_link, version, size_mb):
    try:
        session = sm.get_db_session()

        existing = session.query(BaseModel).filter_by(name=name).first()
        if existing:
            session.close()
            return f"Error: Model '{name}' already exists!", get_base_models_table()

        if not hf_model_id.strip():
            session.close()
            return "Error: HuggingFace Model ID cannot be empty!", get_base_models_table()

        base_model = BaseModel(
            name=name,
            hf_model_id=hf_model_id.strip(),
            model_download_link=download_link.strip() if download_link else None,
            version=version,
            size_mb=int(size_mb) if size_mb else None,
            is_active=True
        )
        session.add(base_model)
        session.commit()
        session.close()

        return f"Model '{name}' added.", get_base_models_table()
    except Exception as e:
        return f"Error: {str(e)}", get_base_models_table()


def update_base_model(model_id, name, hf_model_id, download_link, version, size_mb, is_active):
    if not model_id or not model_id.strip():
        return "Error: Select a model ID to edit.", get_base_models_table()
    try:
        session = sm.get_db_session()
        model = session.query(BaseModel).filter_by(id=model_id.strip()).first()
        if not model:
            session.close()
            return f"Error: Model with ID '{model_id}' not found.", get_base_models_table()

        if name and name.strip():
            model.name = name.strip()
        if hf_model_id and hf_model_id.strip():
            model.hf_model_id = hf_model_id.strip()
        model.model_download_link = download_link.strip() if download_link and download_link.strip() else model.model_download_link
        if version and version.strip():
            model.version = version.strip()
        if size_mb:
            model.size_mb = int(size_mb)
        model.is_active = is_active

        session.commit()
        session.close()
        return f"Model '{model.name}' updated.", get_base_models_table()
    except Exception as e:
        return f"Error: {str(e)}", get_base_models_table()


def delete_base_model(model_id):
    if not model_id or not model_id.strip():
        return "Error: Enter a model ID to delete.", get_base_models_table()
    try:
        session = sm.get_db_session()
        model = session.query(BaseModel).filter_by(id=model_id.strip()).first()
        if not model:
            session.close()
            return f"Error: Model with ID '{model_id}' not found.", get_base_models_table()
        name = model.name
        session.delete(model)
        session.commit()
        session.close()
        return f"Model '{name}' deleted.", get_base_models_table()
    except Exception as e:
        return f"Error: {str(e)}", get_base_models_table()


def load_base_model_for_edit(model_id):
    """Load a model's fields into the edit form."""
    if not model_id or not model_id.strip():
        return "", "", "", "", 0, True
    session = sm.get_db_session()
    try:
        model = session.query(BaseModel).filter_by(id=model_id.strip()).first()
        if not model:
            return "", "", "", "", 0, True
        return (
            model.name or "",
            model.hf_model_id or "",
            model.model_download_link or "",
            model.version or "",
            model.size_mb or 0,
            model.is_active if model.is_active is not None else True,
        )
    finally:
        session.close()


def get_base_models_table():
    session = sm.get_db_session()
    try:
        models = session.query(BaseModel).all()
        data = [{
            'ID': m.id,
            'Name': m.name,
            'HF Model ID': m.hf_model_id or '',
            'Download Link': m.model_download_link or '',
            'Version': m.version,
            'Size (MB)': m.size_mb,
            'Active': 'Yes' if m.is_active else 'No',
        } for m in models]
        return pd.DataFrame(data)
    finally:
        session.close()


def train_new_adapter(
        base_model_id,
        adapter_name,
        domain,
        version,
        dataset_source,
        predefined_dataset,
        custom_file,
        pasted_data,
        epochs,
        batch_size,
        learning_rate,
        quant_type,
):
    """Generator that yields (logs_text, status_markdown) for live UI updates."""
    logs = []
    total_epochs = int(epochs)

    def log(msg):
        logs.append(msg)

    def logs_text():
        return "\n".join(logs)

    def error(msg):
        log(f"ERROR: {msg}")
        return (logs_text(), f"**Error:** {msg}")

    try:
        if not is_gguf_available():
            yield error("Training not available. Check that torch/peft/trl are installed and LLAMA_CPP_DIR is set in .env")
            return

        log(f"Base model ID from dropdown: {base_model_id}")
        log("Loading dataset...")
        yield (logs_text(), "**Status:** Loading dataset...")

        if dataset_source == "Predefined Dataset":
            datasets = get_predefined_datasets()
            data = datasets.get(predefined_dataset, [])
            if not data:
                yield error(f"No data found for '{predefined_dataset}'")
                return

        elif dataset_source == "Upload JSON File":
            if custom_file is None:
                yield error("Please upload a JSON file")
                return
            with open(custom_file.name, 'r') as f:
                data = json.load(f)

        elif dataset_source == "Paste JSON":
            if not pasted_data.strip():
                yield error("Please paste your JSON data")
                return
            data = json.loads(pasted_data)

        else:
            yield error(f"Invalid dataset source: '{dataset_source}'")
            return

        if not isinstance(data, list) or len(data) == 0:
            yield error("Dataset must be a non-empty list")
            return

        log(f"Dataset: {len(data)} examples")

        if not base_model_id:
            yield error("No base model selected. Please select a model from the dropdown.")
            return

        session = sm.get_db_session()
        base_model = session.query(BaseModel).filter_by(id=base_model_id).first()
        session.close()

        if not base_model:
            yield error(f"Base model with ID '{base_model_id}' not found in database.")
            return

        hf_model_id = base_model.hf_model_id
        if not hf_model_id:
            yield error(f"Model '{base_model.name}' has no HuggingFace Model ID. Go to Base Models tab and add one.")
            return

        output_dir = f"./adapters/{domain}_{version}"

        log(f"Initializing trainer with {hf_model_id}...")
        yield (logs_text(), "**Status:** Loading HuggingFace model (this may download on first use)...")

        global gguf_trainer_instance
        try:
            gguf_trainer_instance = GGUFAdapterTrainer(hf_model_id=hf_model_id)
        except Exception as e:
            log(f"ERROR: {e}")
            yield (logs_text(), f"**Error:** {e}")
            return

        log("Model loaded. Starting training...")
        yield (logs_text(), _progress_bar(0, total_epochs) + "\n\n**Status:** Starting training...")

        current_epoch = 0
        result_data = None

        for event, log_line in gguf_trainer_instance.train_adapter_streaming(
            data=data,
            domain=domain,
            adapter_name=adapter_name,
            output_dir=output_dir,
            epochs=total_epochs,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            quant_type=quant_type,
        ):
            evt = event.get("event", "")

            if evt == "error":
                log(log_line)
                yield (logs_text(), f"**Error:** {event.get('message', log_line)}")
                return

            if evt == "result":
                result_data = event["data"]
                log(log_line)
                yield (logs_text(), _progress_bar(total_epochs, total_epochs) + "\n\n**Status:** Training finished!")
                continue

            # Accumulate log
            log(log_line)

            # Build status based on event type
            if evt == "batch":
                epoch = event.get("epoch", current_epoch)
                batch = event.get("batch", 0)
                total = event.get("total", 1)
                loss = event.get("loss", 0)
                phase = event.get("phase", "train")
                status = (
                    _progress_bar(epoch - 1, total_epochs, batch, total)
                    + f"\n\n**Epoch {epoch}/{total_epochs}** | "
                    f"Phase: {phase} | Step: {batch}/{total} | "
                    f"Loss: {loss:.4f}"
                )
                yield (logs_text(), status)
            elif evt == "epoch_complete":
                current_epoch = event.get("epoch", current_epoch + 1)
                status = (
                    _progress_bar(current_epoch, total_epochs)
                    + f"\n\n**Epoch {current_epoch}/{total_epochs} complete** | "
                    f"Train loss: {event.get('train_loss', 0):.4f}"
                )
                yield (logs_text(), status)
            elif evt in ("status", "adapter_created", "data_loaded", "training_start", "complete"):
                yield (logs_text(), _progress_bar(current_epoch, total_epochs) + f"\n\n**Status:** {log_line}")
            else:
                yield (logs_text(), _progress_bar(current_epoch, total_epochs) + f"\n\n**Status:** {log_line}")

        if result_data is None:
            yield (logs_text(), "**Error:** Training ended without result")
            return

        # Upload LoRA adapter to Supabase
        log("Uploading LoRA adapter to Supabase...")
        yield (logs_text(), _progress_bar(total_epochs, total_epochs) + "\n\n**Status:** Uploading LoRA adapter to Supabase...")

        adapter_id = gguf_trainer_instance.package_and_upload(
            adapter_path=result_data['adapter_path'],
            adapter_name=adapter_name,
            domain=domain,
            version=version,
            base_model_id=base_model_id
        )

        log("Deploying adapter...")
        yield (logs_text(), _progress_bar(total_epochs, total_epochs) + "\n\n**Status:** Deploying...")

        gguf_trainer_instance.deploy_adapter(adapter_id, rollout_percentage=100)
        log("LoRA adapter deployed.")

        # Upload base model GGUF via storage provider (HuggingFace)
        base_gguf = result_data.get('base_model_gguf_path', '')
        base_model_url = None
        providers = get_available_providers()

        if base_gguf and providers:
            provider_name = providers[0]
            log(f"Uploading base model GGUF to {provider_name}...")
            yield (logs_text(), _progress_bar(total_epochs, total_epochs) + f"\n\n**Status:** Uploading base model to {provider_name}...")

            try:
                storage = get_storage_provider(provider_name)
                upload_result = storage.upload_model(
                    local_path=base_gguf,
                    model_name=base_model.name,
                )
                base_model_url = upload_result.download_url
                log(f"Base model uploaded: {base_model_url}")

                # Auto-update download link in DB
                session = sm.get_db_session()
                try:
                    db_model = session.query(BaseModel).filter_by(id=base_model_id).first()
                    if db_model:
                        db_model.model_download_link = base_model_url
                        session.commit()
                        log("Download link saved to database.")
                finally:
                    session.close()

            except Exception as upload_err:
                log(f"WARNING: Base model upload failed: {upload_err}")
                log("You can manually upload and set the download link later.")
        elif base_gguf:
            log("No storage provider configured. Skipping base model upload.")
            log("Set HF_TOKEN and HF_REPO_ID in .env to enable auto-upload.")

        log("Done!")

        final_status = f"""**Training Complete**

- Adapter ID: `{adapter_id}`
- LoRA GGUF: `{result_data['adapter_path']}`
- Base Model GGUF: `{base_gguf}`"""

        if base_model_url:
            final_status += f"\n- Base Model URL: `{base_model_url}`"
        else:
            final_status += "\n- Base Model URL: *not uploaded — set HF_TOKEN + HF_REPO_ID in .env*"

        final_status += f"""
- Final Loss: {result_data['final_loss']:.4f}
- Epochs: {result_data['epochs']}
- Status: Deployed (100% rollout)"""

        yield (logs_text(), final_status)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"ERROR: {e}\n{tb}")
        yield (logs_text(), f"**Error during training:**\n\n```\n{e}\n```")


def _progress_bar(epoch, total_epochs, batch=0, total_batches=0):
    """Build a text-based progress bar for display in Markdown."""
    if total_batches > 0:
        epoch_frac = (epoch + batch / total_batches) / total_epochs
    else:
        epoch_frac = epoch / total_epochs if total_epochs > 0 else 0
    pct = min(epoch_frac * 100, 100)
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return f"`[{bar}]` **{pct:.0f}%**"


def toggle_adapter_status(adapter_id, new_status):
    session = sm.get_db_session()
    try:
        adapter = session.query(Adapter).filter_by(id=adapter_id).first()
        if adapter:
            adapter.status = new_status
            session.commit()
            return f"Adapter status updated to {new_status}", get_adapters_dataframe()
        return "Error: Adapter not found", get_adapters_dataframe()
    finally:
        session.close()


def delete_adapter(adapter_id):
    session = sm.get_db_session()
    try:
        adapter = session.query(Adapter).filter_by(id=adapter_id).first()
        if adapter:
            sm.supabase.storage.from_('adapters').remove([adapter.storage_path])
            session.delete(adapter)
            session.commit()
            return "Adapter deleted.", get_adapters_dataframe()
        return "Error: Adapter not found", get_adapters_dataframe()
    finally:
        session.close()


def create_interface():
    with gr.Blocks(title="Adapter Training System") as demo:

        gr.Markdown("# Adapter Training System")
        gr.Markdown("Server-side LoRA SFT training with GGUF adapter export for OTA deployment")

        with gr.Tabs():

            with gr.Tab("Base Models"):
                gr.Markdown("### Model Registry")

                models_table = gr.Dataframe(
                    value=get_base_models_table(),
                    interactive=False
                )
                models_status = gr.Markdown()

                with gr.Tabs():
                    with gr.Tab("Add New"):
                        model_name = gr.Textbox(label="Model Name", placeholder="Qwen-2.5-0.5B-Instruct")
                        hf_model_id = gr.Textbox(
                            label="HuggingFace Model ID (safetensors)",
                            placeholder="Qwen/Qwen2.5-0.5B-Instruct",
                            info="HF repo with safetensors weights — used for LoRA training"
                        )
                        download_link = gr.Textbox(
                            label="Base Model GGUF Download Link (optional)",
                            placeholder="https://huggingface.co/.../model.gguf",
                        )
                        with gr.Row():
                            model_version = gr.Textbox(label="Version", placeholder="1.0.0")
                            model_size = gr.Number(label="Size (MB)", value=700)
                        add_btn = gr.Button("Add Model", variant="primary")

                        add_btn.click(
                            fn=add_base_model,
                            inputs=[model_name, hf_model_id, download_link, model_version, model_size],
                            outputs=[models_status, models_table]
                        )

                    with gr.Tab("Edit / Delete"):
                        with gr.Row():
                            edit_model_id = gr.Textbox(label="Model ID", placeholder="Paste model ID from table above")
                            load_btn = gr.Button("Load", variant="secondary")

                        edit_name = gr.Textbox(label="Model Name")
                        edit_hf_id = gr.Textbox(label="HuggingFace Model ID (safetensors)")
                        edit_download_link = gr.Textbox(label="Base Model GGUF Download Link")
                        with gr.Row():
                            edit_version = gr.Textbox(label="Version")
                            edit_size = gr.Number(label="Size (MB)")
                        edit_active = gr.Checkbox(label="Active", value=True)

                        with gr.Row():
                            save_btn = gr.Button("Save Changes", variant="primary")
                            delete_model_btn = gr.Button("Delete Model", variant="stop")

                        load_btn.click(
                            fn=load_base_model_for_edit,
                            inputs=[edit_model_id],
                            outputs=[edit_name, edit_hf_id, edit_download_link, edit_version, edit_size, edit_active]
                        )

                        save_btn.click(
                            fn=update_base_model,
                            inputs=[edit_model_id, edit_name, edit_hf_id, edit_download_link, edit_version, edit_size, edit_active],
                            outputs=[models_status, models_table]
                        )

                        delete_model_btn.click(
                            fn=delete_base_model,
                            inputs=[edit_model_id],
                            outputs=[models_status, models_table]
                        )

            with gr.Tab("Train Adapter"):
                gr.Markdown("### Train New Adapter")
                gr.Markdown("Trains LoRA via SFT on a safetensors model, then converts **both** the base model and LoRA adapter to GGUF. LoRA GGUF is uploaded to Supabase automatically.")

                with gr.Row():
                    _models = get_base_models()
                    base_model_dropdown = gr.Dropdown(
                        choices=_models,
                        label="Base Model",
                        value=_models[0][1] if _models else None
                    )
                    refresh_models_btn = gr.Button("Refresh")

                with gr.Row():
                    adapter_name_input = gr.Textbox(label="Adapter Name", placeholder="Medical Assistant v1")
                    domain_input = gr.Textbox(label="Domain", placeholder="medical")
                    version_input = gr.Textbox(label="Version", placeholder="1.0.0")

                gr.Markdown("### Dataset")

                dataset_source_radio = gr.Radio(
                    choices=["Predefined Dataset", "Upload JSON File", "Paste JSON"],
                    label="Source",
                    value="Predefined Dataset"
                )

                with gr.Group(visible=True) as predefined_group:
                    predefined_dropdown = gr.Dropdown(
                        choices=list(get_predefined_datasets().keys()),
                        label="Select Dataset",
                        value="Medical"
                    )

                with gr.Group(visible=False) as upload_group:
                    custom_file_input = gr.File(label="Upload File", file_types=[".json"])

                with gr.Group(visible=False) as paste_group:
                    pasted_data_input = gr.TextArea(
                        label="JSON Data",
                        placeholder='[{"instruction": "...", "output": "..."}]',
                        lines=10
                    )

                def update_dataset_inputs(choice):
                    return (
                        gr.update(visible=choice == "Predefined Dataset"),
                        gr.update(visible=choice == "Upload JSON File"),
                        gr.update(visible=choice == "Paste JSON")
                    )

                dataset_source_radio.change(
                    fn=update_dataset_inputs,
                    inputs=[dataset_source_radio],
                    outputs=[predefined_group, upload_group, paste_group]
                )

                gr.Markdown("### Training Configuration")

                with gr.Row():
                    epochs_input = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Epochs")
                    batch_size_input = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size")
                    lr_input = gr.Number(label="Learning Rate", value=2e-4)

                with gr.Row():
                    quant_dropdown = gr.Dropdown(
                        choices=QUANT_TYPES,
                        value="Q4_K_M",
                        label="Base Model Quantization",
                        info="GGUF quant type for the base model. LoRA adapter always uses Q8_0."
                    )

                gr.Markdown("**Backend:** PyTorch + PEFT LoRA + TRL SFTTrainer | Output: quantized GGUF")

                train_btn = gr.Button("Start Training", variant="primary", size="lg")
                train_status = gr.Markdown(value="Ready to train.")
                train_logs = gr.Textbox(
                    label="Training Logs",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                )

                def _train_wrapper(*ui_args):
                    yield from train_new_adapter(*ui_args)

                train_btn.click(
                    fn=_train_wrapper,
                    inputs=[
                        base_model_dropdown,
                        adapter_name_input,
                        domain_input,
                        version_input,
                        dataset_source_radio,
                        predefined_dropdown,
                        custom_file_input,
                        pasted_data_input,
                        epochs_input,
                        batch_size_input,
                        lr_input,
                        quant_dropdown,
                    ],
                    outputs=[train_logs, train_status]
                )

                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=get_base_models()),
                    outputs=[base_model_dropdown]
                )

            with gr.Tab("Manage Adapters"):
                gr.Markdown("### Adapter Registry")

                adapters_table = gr.Dataframe(
                    value=get_adapters_dataframe(),
                    interactive=False
                )

                refresh_adapters_btn = gr.Button("Refresh")

                gr.Markdown("### Operations")

                with gr.Row():
                    adapter_id_input = gr.Textbox(label="Adapter ID", placeholder="Enter adapter ID")
                    new_status_dropdown = gr.Dropdown(
                        choices=["active", "inactive", "deprecated"],
                        label="Status",
                        value="active"
                    )

                with gr.Row():
                    update_status_btn = gr.Button("Update Status")
                    delete_btn = gr.Button("Delete Adapter", variant="stop")

                action_output = gr.Markdown()

                refresh_adapters_btn.click(
                    fn=get_adapters_dataframe,
                    outputs=[adapters_table]
                )

                update_status_btn.click(
                    fn=toggle_adapter_status,
                    inputs=[adapter_id_input, new_status_dropdown],
                    outputs=[action_output, adapters_table]
                )

                delete_btn.click(
                    fn=delete_adapter,
                    inputs=[adapter_id_input],
                    outputs=[action_output, adapters_table]
                )

            with gr.Tab("Deployments"):
                gr.Markdown("### Deployment Management")
                gr.Markdown("Gradual rollouts, A/B testing, and rollback capabilities coming soon.")

        gr.Markdown("---")
        gr.Markdown("Training runs on server (HuggingFace model). GGUF adapters are deployed OTA to devices.")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapter Training System")
    parser.add_argument("--llama-cpp-dir", type=str, help="Path to llama.cpp directory (for convert_lora_to_gguf.py)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")

    args = parser.parse_args()

    # Set LLAMA_CPP_DIR from CLI if provided
    if args.llama_cpp_dir:
        os.environ["LLAMA_CPP_DIR"] = args.llama_cpp_dir

    if is_gguf_available():
        print("Training system ready (torch + peft + trl + convert_lora_to_gguf.py)")
    else:
        print("Warning: Training not fully available.")
        print("  Ensure torch, peft, trl, datasets, accelerate are installed.")
        print("  Set LLAMA_CPP_DIR in .env or via --llama-cpp-dir")

    print(f"Starting on {args.host}:{args.port}...")

    app = create_interface()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )
