import hashlib
import os
import tarfile

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from models import BaseModel, Adapter, AdapterDeployment
from src import default_datasets
from supabase_manager import SupabaseManager


class AdapterTrainer:
    def __init__(self, base_model_name: str = "LiquidAI/LFM2-350M"):
        self.base_model_name = base_model_name
        self.supabase_manager = SupabaseManager()

        # Load base model
        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16,
            device_map="cuda")

    def train_adapter(self, data: list, domain: str, adapter_name: str, output_dir: str, epochs: int = 3):
        """Train a LoRA adapter"""

        # Prepare dataset
        dataset = Dataset.from_list(data)

        def format_data(sample):
            text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
            return {"text": text}

        dataset = dataset.map(format_data)

        # Tokenize
        def tokenize(sample):
            result = self.tokenizer(sample["text"], truncation=True, max_length=512, padding="max_length")
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

        # LoRA config
        lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
            bias="none", task_type=TaskType.CAUSAL_LM, )

        model = get_peft_model(self.base_model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=4,
            save_steps=100, logging_steps=10, learning_rate=2e-4, fp16=True, )

        # Train
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, )

        print(f"Training {adapter_name}...")
        train_result = trainer.train()

        # Save adapter
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return {'output_dir': output_dir, 'final_loss': train_result.training_loss, 'epochs': epochs}

    def package_and_upload(self, adapter_dir: str, adapter_name: str, domain: str, version: str, base_model_id: str):
        """Package adapter and upload to Supabase"""

        # Create tarball
        tar_path = f"{adapter_dir}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(adapter_dir, arcname=os.path.basename(adapter_dir))

        # Calculate checksum
        sha256_hash = hashlib.sha256()
        with open(tar_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        checksum = sha256_hash.hexdigest()

        # Get file size
        size_mb = os.path.getsize(tar_path) / (1024 * 1024)

        # Upload to Supabase Storage
        storage_path = f"{base_model_id}/{domain}/{version}.tar.gz"
        print(f"Uploading to Supabase: {storage_path}")
        self.supabase_manager.upload_adapter(tar_path, storage_path)

        # Save metadata to database
        session = self.supabase_manager.get_db_session()
        try:
            adapter = Adapter(base_model_id=base_model_id, name=adapter_name, domain=domain, version=version,
                storage_path=storage_path, size_mb=round(size_mb, 2), checksum_sha256=checksum, status='active',
                is_published=True)
            session.add(adapter)
            session.commit()

            adapter_id = adapter.id
            print(f"Adapter saved to database with ID: {adapter_id}")

            return adapter_id
        finally:
            session.close()

    def deploy_adapter(self, adapter_id: str, rollout_percentage: int = 100):
        """Deploy adapter for OTA updates"""
        session = self.supabase_manager.get_db_session()
        try:
            deployment = AdapterDeployment(adapter_id=adapter_id, rollout_percentage=rollout_percentage, is_active=True)
            session.add(deployment)
            session.commit()
            print(f"Adapter {adapter_id} deployed with {rollout_percentage}% rollout")
        finally:
            session.close()


# ==== USAGE EXAMPLE ====
if __name__ == "__main__":
    # 1. Initialize trainer
    trainer = AdapterTrainer()

    # 2. First, register base model in DB
    sm = SupabaseManager()
    session = sm.get_db_session()

    base_model = BaseModel(name="LFM2-350M", huggingface_id="LiquidAI/LFM2-350M", version="1.0.0", size_mb=700,
        is_active=True)
    session.add(base_model)
    session.commit()
    base_model_id = base_model.id
    session.close()

    # 3. Prepare training data
    medical_data = default_datasets.medical_data

    # 4. Train adapter
    result = trainer.train_adapter(data=medical_data, domain="medical", adapter_name="Medical Adapter",
        output_dir="./lora_adapter_medical", epochs=3)

    # 5. Package and upload
    adapter_id = trainer.package_and_upload(adapter_dir="./lora_adapter_medical", adapter_name="Medical Adapter",
        domain="medical", version="1.0.0", base_model_id=base_model_id)

    # 6. Deploy for OTA
    trainer.deploy_adapter(adapter_id, rollout_percentage=100)
