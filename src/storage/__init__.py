from .base import StorageProvider, UploadResult
from .huggingface import HuggingFaceStorage

# Registry of available storage providers.
# Add new providers here (e.g., S3Storage, GCSStorage).
STORAGE_PROVIDERS = {
    "HuggingFace": HuggingFaceStorage,
}


def get_storage_provider(name: str) -> StorageProvider:
    """Get a storage provider instance by name."""
    cls = STORAGE_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown storage provider: '{name}'. "
            f"Available: {list(STORAGE_PROVIDERS.keys())}"
        )
    return cls()


def get_available_providers() -> list[str]:
    """Return names of providers that are configured and ready."""
    available = []
    for name, cls in STORAGE_PROVIDERS.items():
        try:
            instance = cls()
            if instance.is_configured():
                available.append(name)
        except Exception:
            pass
    return available


__all__ = [
    'StorageProvider',
    'UploadResult',
    'HuggingFaceStorage',
    'STORAGE_PROVIDERS',
    'get_storage_provider',
    'get_available_providers',
]
