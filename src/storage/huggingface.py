import os
import logging
from pathlib import Path
from typing import Optional, Callable

from huggingface_hub import HfApi, hf_hub_url

from .base import StorageProvider, UploadResult

logger = logging.getLogger(__name__)


class HuggingFaceStorage(StorageProvider):
    """Upload GGUF models to a HuggingFace model repo.

    Config via env vars:
        HF_TOKEN    - HuggingFace access token (write permission)
        HF_REPO_ID  - Target repo, e.g. "myuser/adapter-models"

    Repo structure:
        {model_name}/base-model-Q4_K_M.gguf
        {model_name}/adapters/medical_v1_lora.gguf
    """

    def __init__(self):
        self.token = os.environ.get("HF_TOKEN", "")
        self.repo_id = os.environ.get("HF_REPO_ID", "")
        self._api = None

    @property
    def api(self) -> HfApi:
        if self._api is None:
            self._api = HfApi(token=self.token)
        return self._api

    def is_configured(self) -> bool:
        return bool(self.token) and bool(self.repo_id)

    def _ensure_repo(self):
        """Create the repo if it doesn't exist."""
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            exist_ok=True,
        )

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> UploadResult:
        if not self.is_configured():
            raise RuntimeError(
                "HuggingFace storage not configured. "
                "Set HF_TOKEN and HF_REPO_ID in .env"
            )

        self._ensure_repo()

        local_file = Path(local_path)
        size_bytes = local_file.stat().st_size
        logger.info(
            f"Uploading {local_file.name} ({size_bytes / 1024 / 1024:.1f} MB) "
            f"to {self.repo_id}/{remote_path}"
        )

        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="model",
        )

        download_url = hf_hub_url(
            repo_id=self.repo_id,
            filename=remote_path,
            repo_type="model",
        )

        logger.info(f"Uploaded: {download_url}")
        return UploadResult(
            download_url=download_url,
            remote_path=remote_path,
            size_bytes=size_bytes,
        )

    def delete_file(self, remote_path: str) -> None:
        if not self.is_configured():
            return
        try:
            self.api.delete_file(
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="model",
            )
            logger.info(f"Deleted: {self.repo_id}/{remote_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {remote_path}: {e}")

    def upload_model(
        self,
        local_path: str,
        model_name: str,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> UploadResult:
        """Upload a base model GGUF. Stored as {model_name}/{filename}."""
        filename = Path(local_path).name
        remote_path = f"{model_name}/{filename}"
        return self.upload_file(local_path, remote_path, on_progress)

    def upload_adapter(
        self,
        local_path: str,
        model_name: str,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> UploadResult:
        """Upload a LoRA adapter GGUF. Stored as {model_name}/adapters/{filename}."""
        filename = Path(local_path).name
        remote_path = f"{model_name}/adapters/{filename}"
        return self.upload_file(local_path, remote_path, on_progress)
