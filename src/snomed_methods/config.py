#!/usr/bin/env python3
import os
from typing import Optional


class SnomedConfig:
    _instance: Optional["SnomedConfig"] = None

    def __new__(cls) -> "SnomedConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.uk_snomed_path: str = os.environ.get(
            "SNOMED_UK_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "uk_sct2cl_42.2.0",
                "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            ),
        )
        self.embedding_model_path: str = os.environ.get("SNOMED_EMBEDDING_MODEL", None)
        self.embedding_backend: str = os.environ.get("EMBEDDER_BACKEND", "hf")
        self.snowstorm_api_url: str = os.environ.get(
            "SNOWSTREAM_API_URL", "https://snowstorm.ihtsdotools.org"
        )
        self.ollama_base_url: str = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.debug_mode: bool = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        self.verify_ssl: bool = os.environ.get("VERIFY_SSL", "true").lower() != "false"
        self.default_top_k: int = 20
        self.max_concepts: int = 100
        self.batch_size: int = 64
        self.medcat_model_pack_path: str = os.environ.get(
            "MEDCAT_MODEL_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "model_packs",
                "medcat_model_pack_422d1d38fc58f158.zip",
            ),
        )
        self.cache_dir: str = os.environ.get("SNOMED_CACHE_DIR", "./.cache")
        self.embeddings_cache_dir: str = os.path.join(self.cache_dir, "embeddings")
        self.embedding_format: str = os.environ.get("EMBEDDING_FORMAT", "numpy")
        self.use_pickle: bool = os.environ.get("USE_PICKLE", "false").lower() == "true"
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "SnomedConfig":
        return cls()

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance = None


def get_config() -> SnomedConfig:
    return SnomedConfig.get_instance()


ENV_FILES = [".env", ".env.local"]


def _load_env_files() -> None:
    for env_file in ENV_FILES:
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key.strip(), value.strip())


SECRETS_FILES = [".secrets", ".secrets.local"]


def _load_secrets() -> None:
    for secrets_file in SECRETS_FILES:
        if os.path.exists(secrets_file):
            with open(secrets_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key.strip(), value.strip())


_load_env_files()
_load_secrets()
