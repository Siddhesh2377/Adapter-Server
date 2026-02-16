from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class UploadResult:
    download_url: str
    remote_path: str
    size_bytes: int


class StorageProvider(ABC):
    """Base class for model storage plugins.

    To add a new storage backend (S3, GCS, etc.):
      1. Subclass StorageProvider
      2. Implement upload_file() and delete_file()
      3. Register in src/storage/__init__.py
    """

    @abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> UploadResult:
        """Upload a file and return its public download URL."""
        ...

    @abstractmethod
    def delete_file(self, remote_path: str) -> None:
        """Delete a previously uploaded file."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this provider has all required config (tokens, etc.)."""
        ...
