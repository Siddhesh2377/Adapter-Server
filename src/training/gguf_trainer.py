import json
import os
import sys
import logging
import subprocess
import threading
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

from src.models.models import Adapter, AdapterDeployment
from src.supabase_manager import SupabaseManager

logger = logging.getLogger(__name__)


def _get_llama_cpp_dir() -> Path:
    """Get llama.cpp directory from LLAMA_CPP_DIR env var."""
    llama_dir = os.environ.get("LLAMA_CPP_DIR", "")
    return Path(llama_dir) if llama_dir else Path("")


def _get_convert_lora_script() -> Path:
    """Find convert_lora_to_gguf.py from LLAMA_CPP_DIR."""
    script = _get_llama_cpp_dir() / "convert_lora_to_gguf.py"
    return script if script.exists() else Path("")


def _get_convert_hf_script() -> Path:
    """Find convert_hf_to_gguf.py from LLAMA_CPP_DIR."""
    script = _get_llama_cpp_dir() / "convert_hf_to_gguf.py"
    return script if script.exists() else Path("")


def _get_quantize_binary() -> Path:
    """Find llama-quantize binary from LLAMA_CPP_DIR/build/bin/."""
    binary = _get_llama_cpp_dir() / "build" / "bin" / "llama-quantize"
    return binary if binary.exists() else Path("")


# Quantization types available in the UI (no f16/f32)
QUANT_TYPES = [
    "Q2_K",
    "Q3_K_M",
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
]


def is_gguf_available() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
    except ImportError:
        return False
    return (
        _get_convert_lora_script().exists()
        and _get_convert_hf_script().exists()
        and _get_quantize_binary().exists()
    )


class _StreamingCallback(TrainerCallback):
    """Pushes training events into a list so the generator can yield them."""

    def __init__(self, event_queue: list, lock: threading.Lock, total_epochs: int):
        self.events = event_queue
        self.lock = lock
        self.total_epochs = total_epochs

    def _push(self, event):
        with self.lock:
            self.events.append(event)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        epoch = logs.get("epoch", 0)
        lr = logs.get("learning_rate", 0)
        step = state.global_step
        total = state.max_steps
        self._push({
            "event": "batch",
            "epoch": int(epoch) + 1,
            "batch": step,
            "total": total,
            "loss": round(loss, 4),
            "lr": lr,
            "phase": "train",
        })

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        logs = state.log_history
        # Find latest loss for this epoch
        train_loss = 0.0
        for entry in reversed(logs):
            if "loss" in entry:
                train_loss = entry["loss"]
                break
        self._push({
            "event": "epoch_complete",
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._push({"event": "train_end"})


class GGUFAdapterTrainer:
    def __init__(self, hf_model_id: str):
        if not is_gguf_available():
            raise RuntimeError(
                "Training dependencies not available.\n"
                "Install: pip install torch transformers peft trl datasets accelerate\n"
                "Set LLAMA_CPP_DIR in .env to point to your llama.cpp directory."
            )

        self.hf_model_id = hf_model_id
        self.supabase_manager = SupabaseManager()
        self.convert_lora_script = _get_convert_lora_script()
        self.convert_hf_script = _get_convert_hf_script()
        self.quantize_binary = _get_quantize_binary()

        # Download model snapshot once — used for both training and GGUF conversion
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading model snapshot: {hf_model_id}")
        self.local_model_path = snapshot_download(repo_id=hf_model_id)
        logger.info(f"Model cached at: {self.local_model_path}")

        logger.info("Loading tokenizer and model from local snapshot...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("Model loaded")

    @staticmethod
    def _format_training_data(data: List[Dict]) -> Dataset:
        texts = []
        for item in data:
            text = f"### Instruction:\n{item['instruction']}\n### Response:\n{item['output']}"
            texts.append(text)
        return Dataset.from_dict({"text": texts})

    def train_adapter_streaming(
        self,
        data: List[Dict],
        domain: str,
        adapter_name: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 0.0001,
        rank: int = 8,
        alpha: float = 32.0,
        quant_type: str = "Q4_K_M",
    ):
        """Generator that yields (event_dict, log_line) tuples as training progresses."""
        os.makedirs(output_dir, exist_ok=True)

        yield ({"event": "status"}, f"Formatting {len(data)} examples...")
        dataset = self._format_training_data(data)
        yield ({"event": "data_loaded", "count": len(data)}, f"Dataset ready: {len(data)} examples")

        # LoRA config
        yield ({"event": "status"}, f"Applying LoRA (rank={rank}, alpha={alpha})...")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        yield (
            {"event": "adapter_created", "rank": rank, "alpha": alpha},
            f"LoRA applied: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)"
        )

        # Training args
        peft_output_dir = os.path.join(output_dir, "peft_adapter")
        training_args = TrainingArguments(
            output_dir=peft_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )

        # Streaming callback
        event_queue = []
        lock = threading.Lock()
        callback = _StreamingCallback(event_queue, lock, epochs)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            callbacks=[callback],
        )

        yield (
            {"event": "training_start", "epochs": epochs, "lr": learning_rate},
            f"Training started: {epochs} epochs, lr={learning_rate}, batch_size={batch_size}"
        )

        # Run training in a thread so we can yield events
        train_result = [None]
        train_error = [None]

        def _run_training():
            try:
                train_result[0] = trainer.train()
            except Exception as e:
                train_error[0] = e

        train_thread = threading.Thread(target=_run_training)
        train_thread.start()

        while train_thread.is_alive():
            train_thread.join(timeout=0.5)
            with lock:
                pending = list(event_queue)
                event_queue.clear()
            for event in pending:
                evt = event.get("event")
                if evt == "batch":
                    msg = (
                        f"[{event['phase']}] step {event['batch']}/{event['total']} "
                        f"loss={event['loss']:.4f}"
                    )
                    yield (event, msg)
                elif evt == "epoch_complete":
                    msg = f"Epoch {event['epoch']} complete: train_loss={event['train_loss']:.4f}"
                    yield (event, msg)

        # Drain remaining events
        with lock:
            pending = list(event_queue)
            event_queue.clear()
        for event in pending:
            evt = event.get("event")
            if evt == "batch":
                msg = f"[{event['phase']}] step {event['batch']}/{event['total']} loss={event['loss']:.4f}"
                yield (event, msg)
            elif evt == "epoch_complete":
                msg = f"Epoch {event['epoch']} complete: train_loss={event['train_loss']:.4f}"
                yield (event, msg)

        if train_error[0]:
            yield ({"event": "error", "message": str(train_error[0])}, f"ERROR: {train_error[0]}")
            return

        # Get final loss
        final_loss = 0.0
        if train_result[0] and hasattr(train_result[0], "training_loss"):
            final_loss = train_result[0].training_loss

        # Save PEFT adapter
        yield ({"event": "status"}, "Saving PEFT adapter...")
        model.save_pretrained(peft_output_dir)
        self.tokenizer.save_pretrained(peft_output_dir)
        yield ({"event": "status"}, f"PEFT adapter saved to {peft_output_dir}")

        # Convert LoRA adapter to GGUF (q8_0 — small file, needs precision)
        adapter_filename = f"{domain}_{adapter_name.replace(' ', '_')}_lora.gguf"
        adapter_path = os.path.join(output_dir, adapter_filename)

        yield ({"event": "status"}, "Converting LoRA adapter to GGUF (Q8_0)...")
        lora_cmd = [
            sys.executable,
            str(self.convert_lora_script),
            "--base-model-id", self.hf_model_id,
            "--outfile", adapter_path,
            "--outtype", "q8_0",
            peft_output_dir,
        ]
        logger.info(f"Running: {' '.join(lora_cmd)}")

        result = subprocess.run(lora_cmd, capture_output=True)
        if result.returncode != 0:
            err_msg = (result.stderr or result.stdout or b"Unknown error").decode("utf-8", errors="replace")
            yield ({"event": "error", "message": err_msg}, f"LoRA GGUF conversion failed:\n{err_msg}")
            return

        yield ({"event": "status"}, f"LoRA GGUF saved: {adapter_path}")

        # Convert base model to GGUF — first to f16, then quantize
        safe_name = self.hf_model_id.replace("/", "_")
        base_f16_path = os.path.join(output_dir, f"{safe_name}-f16.gguf")
        base_model_gguf_path = os.path.join(output_dir, f"{safe_name}-{quant_type}.gguf")

        yield ({"event": "status"}, "Converting base model to GGUF (f16 intermediate)...")
        hf_cmd = [
            sys.executable,
            str(self.convert_hf_script),
            self.local_model_path,
            "--outfile", base_f16_path,
            "--outtype", "f16",
        ]
        logger.info(f"Running: {' '.join(hf_cmd)}")

        result = subprocess.run(hf_cmd, capture_output=True)
        if result.returncode != 0:
            err_msg = (result.stderr or result.stdout or b"Unknown error").decode("utf-8", errors="replace")
            yield ({"event": "error", "message": err_msg}, f"Base model GGUF conversion failed:\n{err_msg}")
            return

        # Quantize base model to target type
        yield ({"event": "status"}, f"Quantizing base model to {quant_type}...")
        quant_cmd = [
            str(self.quantize_binary),
            base_f16_path,
            base_model_gguf_path,
            quant_type,
        ]
        logger.info(f"Running: {' '.join(quant_cmd)}")

        result = subprocess.run(quant_cmd, capture_output=True)
        if result.returncode != 0:
            err_msg = (result.stderr or result.stdout or b"Unknown error").decode("utf-8", errors="replace")
            yield ({"event": "error", "message": err_msg}, f"Quantization failed:\n{err_msg}")
            return

        # Clean up f16 intermediate
        try:
            os.remove(base_f16_path)
        except OSError:
            pass

        yield ({"event": "status"}, f"Base model GGUF saved: {base_model_gguf_path} ({quant_type})")

        yield ({"event": "complete", "output": adapter_path, "final_train_loss": final_loss}, "Training complete!")
        yield ({"event": "result", "data": {
            "output_dir": output_dir,
            "adapter_path": adapter_path,
            "base_model_gguf_path": base_model_gguf_path,
            "final_loss": final_loss,
            "epochs": epochs,
            "format": "gguf",
        }}, "Done")

    def train_adapter(
        self,
        data: List[Dict],
        domain: str,
        adapter_name: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 0.0001,
    ) -> Dict:
        """Non-streaming wrapper."""
        result = None
        for event, log_line in self.train_adapter_streaming(
            data, domain, adapter_name, output_dir, epochs, batch_size, learning_rate
        ):
            logger.info(log_line)
            if event.get("event") == "result":
                result = event["data"]
            elif event.get("event") == "error":
                raise RuntimeError(event.get("message", "Training failed"))
        if result is None:
            raise RuntimeError("Training ended without result")
        return result

    def package_and_upload(
        self,
        adapter_path: str,
        adapter_name: str,
        domain: str,
        version: str,
        base_model_id: str,
    ) -> str:
        import tarfile
        import hashlib

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter GGUF not found: {adapter_path}")
        logger.info(f"Packaging adapter: {adapter_path}")

        tar_path = f"{adapter_path}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(adapter_path, arcname=os.path.basename(adapter_path))

        sha256_hash = hashlib.sha256()
        with open(tar_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        checksum = sha256_hash.hexdigest()

        size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        logger.info(f"Adapter size: {size_mb:.2f} MB, Checksum: {checksum[:16]}...")

        storage_path = f"{base_model_id}/{domain}/{version}.tar.gz"
        logger.info(f"Uploading to Supabase: {storage_path}")
        self.supabase_manager.upload_adapter(tar_path, storage_path)

        session = self.supabase_manager.get_db_session()
        try:
            adapter = Adapter(
                base_model_id=base_model_id,
                name=adapter_name,
                domain=domain,
                version=version,
                storage_path=storage_path,
                size_mb=round(size_mb, 2),
                checksum_sha256=checksum,
                status="active",
                is_published=True,
            )
            session.add(adapter)
            session.commit()
            adapter_id = adapter.id
            logger.info(f"Adapter saved to database with ID: {adapter_id}")
            return adapter_id
        finally:
            session.close()

    def deploy_adapter(self, adapter_id: str, rollout_percentage: int = 100):
        logger.info(f"Deploying adapter {adapter_id} with {rollout_percentage}% rollout")

        session = self.supabase_manager.get_db_session()
        try:
            deployment = AdapterDeployment(
                adapter_id=adapter_id,
                rollout_percentage=rollout_percentage,
                is_active=True,
            )
            session.add(deployment)
            session.commit()
            logger.info("Deployment record created")
        finally:
            session.close()

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
