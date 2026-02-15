import json
import os

import gradio as gr
import pandas as pd

from src import default_datasets
from src.models.models import BaseModel, Adapter
from supabase_manager import SupabaseManager
from trainer_pipeline import AdapterTrainer

# Initialize managers
sm = SupabaseManager()
trainer_instance = None  # We'll initialize this when user selects a model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_base_models():
    """Fetch all base models from DB"""
    session = sm.get_db_session()
    try:
        models = session.query(BaseModel).all()
        return [(f"{m.name} ({m.version})", m.id) for m in models]
    finally:
        session.close()


def get_adapters_dataframe():
    """Fetch all adapters as a pandas DataFrame"""
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
    """Get list of predefined datasets"""
    return {
        "Medical": default_datasets.medical_data,
        "Coding": default_datasets.coding_data if hasattr(default_datasets, 'coding_data') else [],
        "Creative": default_datasets.creative_data if hasattr(default_datasets, 'creative_data') else [],
        "General": default_datasets.general_data if hasattr(default_datasets, 'general_data') else [],
    }


# ============================================================================
# TAB 1: BASE MODEL MANAGEMENT
# ============================================================================

def add_base_model(name, hf_id, version, size_mb):
    """Add a new base model to the database"""
    try:
        session = sm.get_db_session()

        # Check if model already exists
        existing = session.query(BaseModel).filter_by(name=name).first()
        if existing:
            session.close()
            return f"Error: Model '{name}' already exists!", get_base_models_table()

        base_model = BaseModel(
            name=name,
            huggingface_id=hf_id,
            version=version,
            size_mb=int(size_mb) if size_mb else None,
            is_active=True
        )
        session.add(base_model)
        session.commit()
        session.close()

        return f"Success: Model '{name}' added successfully!", get_base_models_table()
    except Exception as e:
        return f"Error: {str(e)}", get_base_models_table()


def get_base_models_table():
    """Get base models as DataFrame for display"""
    session = sm.get_db_session()
    try:
        models = session.query(BaseModel).all()
        data = [{
            'ID': m.id,
            'Name': m.name,
            'HuggingFace ID': m.huggingface_id,
            'Version': m.version,
            'Size (MB)': m.size_mb,
            'Active': 'Yes' if m.is_active else 'No',
            'Created': m.created_at.strftime('%Y-%m-%d')
        } for m in models]
        return pd.DataFrame(data)
    finally:
        session.close()


# ============================================================================
# TAB 2: TRAIN NEW ADAPTER (MAIN FEATURE)
# ============================================================================

def train_new_adapter(
        base_model_id,
        adapter_name,
        domain,
        version,
        dataset_source,  # "predefined" or "custom" or "paste"
        predefined_dataset,
        custom_file,
        pasted_data,
        epochs,
        batch_size,
        learning_rate,
        progress=gr.Progress()
):
    """Main training function with progress tracking"""

    try:
        # Step 1: Determine dataset
        progress(0.1, desc="Loading dataset...")

        if dataset_source == "Predefined Dataset":
            datasets = get_predefined_datasets()
            data = datasets.get(predefined_dataset, [])
            if not data:
                return f"Error: No data found for {predefined_dataset}"

        elif dataset_source == "Upload JSON File":
            if custom_file is None:
                return "Error: Please upload a JSON file"
            with open(custom_file.name, 'r') as f:
                data = json.load(f)

        elif dataset_source == "Paste JSON":
            if not pasted_data.strip():
                return "Error: Please paste your JSON data"
            data = json.loads(pasted_data)

        else:
            return "Error: Invalid dataset source"

        if not isinstance(data, list) or len(data) == 0:
            return "Error: Dataset must be a non-empty list"

        # Step 2: Initialize trainer
        progress(0.2, desc="Loading base model...")

        session = sm.get_db_session()
        base_model = session.query(BaseModel).filter_by(id=base_model_id).first()
        session.close()

        if not base_model:
            return "Error: Base model not found"

        global trainer_instance
        if trainer_instance is None or trainer_instance.base_model_name != base_model.huggingface_id:
            trainer_instance = AdapterTrainer(base_model.huggingface_id)

        # Step 3: Train adapter
        progress(0.3, desc=f"Training adapter (0/{epochs} epochs)...")

        output_dir = f"./adapters/{domain}_{version}"
        os.makedirs(output_dir, exist_ok=True)

        result = trainer_instance.train_adapter(
            data=data,
            domain=domain,
            adapter_name=adapter_name,
            output_dir=output_dir,
            epochs=int(epochs)
        )

        # Step 4: Package and upload
        progress(0.8, desc="Uploading to Supabase...")

        adapter_id = trainer_instance.package_and_upload(
            adapter_dir=output_dir,
            adapter_name=adapter_name,
            domain=domain,
            version=version,
            base_model_id=base_model_id
        )

        # Step 5: Deploy
        progress(0.95, desc="Deploying adapter...")

        trainer_instance.deploy_adapter(adapter_id, rollout_percentage=100)

        progress(1.0, desc="Complete!")

        return f"""**Training Complete**

- Adapter ID: `{adapter_id}`
- Final Loss: {result['final_loss']:.4f}
- Epochs: {result['epochs']}
- Status: Deployed (100% rollout)

Your adapter is now live and available for OTA updates."""

    except Exception as e:
        return f"**Error during training:**\n\n```\n{str(e)}\n```"


# ============================================================================
# TAB 3: ADAPTER MANAGEMENT
# ============================================================================

def toggle_adapter_status(adapter_id, new_status):
    """Activate or deactivate an adapter"""
    session = sm.get_db_session()
    try:
        adapter = session.query(Adapter).filter_by(id=adapter_id).first()
        if adapter:
            adapter.status = new_status
            session.commit()
            return f"Success: Adapter status updated to {new_status}", get_adapters_dataframe()
        return "Error: Adapter not found", get_adapters_dataframe()
    finally:
        session.close()


def delete_adapter(adapter_id):
    """Delete an adapter (with confirmation)"""
    session = sm.get_db_session()
    try:
        adapter = session.query(Adapter).filter_by(id=adapter_id).first()
        if adapter:
            # Also delete from storage
            sm.supabase.storage.from_('adapters').remove([adapter.storage_path])

            # Delete from DB (cascades to deployments)
            session.delete(adapter)
            session.commit()
            return f"Success: Adapter deleted successfully", get_adapters_dataframe()
        return "Error: Adapter not found", get_adapters_dataframe()
    finally:
        session.close()


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    with gr.Blocks(title="Adapter Training System") as app:

        gr.Markdown("# Adapter Training System")
        gr.Markdown("Enterprise-grade LoRA adapter training and deployment platform")

        with gr.Tabs():

            # BASE MODEL MANAGEMENT
            with gr.Tab("Base Models"):
                gr.Markdown("### Model Registry")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Add New Model")
                        model_name = gr.Textbox(label="Model Name", placeholder="Qwen-2.5")
                        hf_id = gr.Textbox(label="HuggingFace ID", placeholder="LiquidAI/LFM2-350M")
                        model_version = gr.Textbox(label="Version", placeholder="1.0.0")
                        model_size = gr.Number(label="Size (MB)", value=700)
                        add_btn = gr.Button("Add Model", variant="primary")
                        add_status = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("#### Existing Models")
                        models_table = gr.Dataframe(
                            value=get_base_models_table(),
                            interactive=False
                        )

                add_btn.click(
                    fn=add_base_model,
                    inputs=[model_name, hf_id, model_version, model_size],
                    outputs=[add_status, models_table]
                )

            # TRAIN NEW ADAPTER
            with gr.Tab("Train Adapter"):
                gr.Markdown("### Train New Adapter")

                with gr.Row():
                    base_model_dropdown = gr.Dropdown(
                        choices=[m[0] for m in get_base_models()],
                        label="Base Model",
                        value=get_base_models()[0][0] if get_base_models() else None
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

                train_btn = gr.Button("Start Training", variant="primary", size="lg")
                train_output = gr.Markdown()

                def get_model_id_from_selection(selection):
                    models = get_base_models()
                    for name, model_id in models:
                        if name == selection:
                            return model_id
                    return None

                train_btn.click(
                    fn=lambda *args: train_new_adapter(
                        get_model_id_from_selection(args[0]),
                        *args[1:]
                    ),
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
                        lr_input
                    ],
                    outputs=[train_output]
                )

                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=[m[0] for m in get_base_models()]),
                    outputs=[base_model_dropdown]
                )

            # ADAPTER MANAGEMENT
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

            # DEPLOYMENTS
            with gr.Tab("Deployments"):
                gr.Markdown("### Deployment Management")
                gr.Markdown(
                    "Advanced deployment features including gradual rollouts, A/B testing, and rollback capabilities will be available soon.")

        gr.Markdown("---")
        gr.Markdown(
            "Trained adapters are automatically deployed with 100% rollout. Connected devices will receive updates on next synchronization.")

    return app


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )