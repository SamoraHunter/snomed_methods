#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Clinical Concept Embedder Module.

Generates high-dimensional vector embeddings for SNOMED CT / MedCAT Concept
Databases using open-source LLMs or clinical transformers.
"""

from __future__ import annotations

import logging
import os
import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import tqdm

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel: Any = None  # type: ignore[assignment]
    AutoTokenizer: Any = None  # type: ignore[assignment]

try:
    import ollama
except ImportError:
    ollama = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]


class ClinicalConceptEmbedder:
    """Generate embeddings for clinical concepts using various LLM backends."""

    def __init__(  # noqa: PLR0913
        self,
        model_name_or_path: str,
        *,
        backend: str = "hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
        ollama_base_url: str = "http://localhost:11434",
        transformers_model_type: str | None = None,
    ) -> None:
        """Initialize the embedder with a model and backend.

        Args:
            model_name_or_path: Model identifier (e.g., 'all-MiniLM-L6-v2')
                               or 'qwen2.5-coder' for Ollama.
                               Can be a local path to a Transformers model directory.
            backend: 'hf' for Hugging Face SentenceTransformer, 'transformers' for
                direct HF Transformers models (e.g., SapBERT), 'ollama' for Ollama API.
            device: Device for inference ('cuda', 'cpu').
            batch_size: Batch size for embedding generation.
            ollama_base_url: Base URL for Ollama instance (default: http://localhost:11434).
            transformers_model_type: Optional model type for Transformers backend
                (e.g., 'BertModel'). If None, auto-inferred.

        """
        self.model_name_or_path: str = model_name_or_path
        self.backend: str = backend
        self.device: str = device
        self.batch_size: int = batch_size
        self.ollama_base_url: str = ollama_base_url
        self.transformers_model_type: str | None = transformers_model_type

        self._logger = logging.getLogger(__name__)

        if backend == "hf":
            self._init_hf_model()
        elif backend == "transformers":
            self._init_transformers_model()
        elif backend == "ollama":
            self._init_ollama_client(self.ollama_base_url)
        else:
            msg = (
                f"Unsupported backend: {backend}. Use 'hf', 'transformers', or 'ollama'"
            )
            raise ValueError(
                msg,
            )

    def _init_hf_model(self) -> None:
        """Initialize Hugging Face SentenceTransformer."""
        try:
            self.model = SentenceTransformer(
                self.model_name_or_path,
                device=self.device,
            )
            self._model_type = "hf"
        except (FileNotFoundError, PermissionError, OSError) as e:
            msg = f"Failed to load HF model: {e}"
            raise RuntimeError(msg) from None

    def _init_transformers_model(self) -> None:
        """Initialize Hugging Face Transformers model directly."""
        if AutoModel is None or AutoTokenizer is None:
            msg = "Transformers package not installed. Run: pip install transformers"
            raise RuntimeError(
                msg,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,
        ).to(self.device)
        self._model_type = "transformers"

    def _init_ollama_client(self, base_url: str) -> None:
        """Initialize Ollama client with custom URL.

        Args:
            base_url: Base URL for Ollama instance (e.g., http://localhost:11434).

        """
        if ollama is None:
            msg = "Ollama package not installed. Run: pip install ollama"
            raise RuntimeError(
                msg,
            )

        self.ollama = ollama
        os.environ["OLLAMA_HOST"] = base_url
        self._model_type = "ollama"

    def prepare_concept_text(
        self,
        concept_df: pd.DataFrame,
        text_columns: list[str] | None = None,
    ) -> list[str]:
        """Format concept data into dense clinical text prompts.

        Args:
            concept_df: DataFrame with concept information (cui, preferred_name,
                       synonyms, type_id columns).
            text_columns: Column names to include in prompt. Defaults to
                         ['preferred_name', 'synonyms', 'type_id'].

        Returns:
            List of formatted concept descriptions.

        """
        if text_columns is None:
            text_columns = ["preferred_name", "synonyms", "type_id"]

        descriptions = []

        for _idx, row in concept_df.iterrows():
            parts = []

            cui = str(row.get("cui", row.get("concept_id", "")))
            parts.append(f"Concept ID: {cui}")

            if "preferred_name" in row and pd.notna(row["preferred_name"]):
                pref_name = str(row["preferred_name"]).strip()
                parts.append(f"Preferred Term: {pref_name}")

            if "synonyms" in row and pd.notna(row["synonyms"]):
                synonyms = str(row["synonyms"]).strip()
                if synonyms:
                    parts.append(f"Synonyms: {synonyms}")

            if "type_id" in row and pd.notna(row.get("type_id")):
                type_id = str(row["type_id"]).strip()
                if type_id:
                    parts.append(f"Semantic Tag: {type_id}")

            description = " | ".join(parts)
            descriptions.append(description)

        return descriptions

    @torch.no_grad()
    def _load_checkpoint(
        self,
        checkpoint_path: str | None,
        concept_texts_length: int,
    ) -> tuple[list, int]:
        """Load embeddings from checkpoint if available.

        Args:
            checkpoint_path: Path to checkpoint file.
            concept_texts_length: Total number of concepts to process.

        Returns:
            Tuple of (embeddings_list, processed_count).

        """
        if not checkpoint_path or not pathlib.Path(checkpoint_path).exists():
            return [], 0

        try:
            with pathlib.Path(checkpoint_path).open("rb") as f:
                saved_data = pickle.load(f)  # noqa: S301
            all_embeddings = saved_data.get("embeddings", [])
            processed_count = len(all_embeddings)
            logger.info(
                "Resuming from checkpoint: %d / %d concepts",
                processed_count,
                concept_texts_length,
            )
        except (FileNotFoundError, PermissionError, OSError):
            logger.warning("Failed to load checkpoint, starting fresh")
            all_embeddings = []
            processed_count = 0
        return all_embeddings, processed_count

    def _embed_batch_hf(self, batch_texts: list[str]) -> list:
        """Embed a batch using Hugging Face SentenceTransformer.

        Args:
            batch_texts: List of text to embed.

        Returns:
            List of embeddings.

        """
        batch_embeddings = self.model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return list(batch_embeddings)

    def _embed_batch_transformers(self, batch_texts: list[str]) -> list:
        """Embed a batch using Hugging Face Transformers.

        Args:
            batch_texts: List of text to embed.

        Returns:
            List of embeddings.

        """
        toks = self.tokenizer.batch_encode_plus(
            batch_texts,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        toks_device: dict[str, torch.Tensor] = {}
        for k, v in toks.items():
            toks_device[k] = v.to(self.device)

        with torch.no_grad():
            output = self.model(**toks_device)
            cls_rep = output[0][:, 0, :]
            batch_embeddings = cls_rep.cpu().numpy()
            return list(batch_embeddings)

    def _embed_batch_ollama(self, batch_texts: list[str]) -> list:
        """Embed a batch using Ollama API.

        Args:
            batch_texts: List of text to embed.

        Returns:
            List of embeddings.

        """
        try:
            batch_embeddings_list = [
                np.array(
                    self.ollama.embeddings(model=self.model_name_or_path, prompt=text)[
                        "embedding"
                    ],
                )
                for text in batch_texts
            ]
        except Exception:
            logger.exception("Error embedding batch. Skipping.")
            batch_embeddings_list = []

        return batch_embeddings_list

    def generate_embeddings(
        self,
        concept_texts: list[str],
        batch_size: int | None = None,
        checkpoint_path: str | None = None,
        checkpoint_interval: int = 500,
    ) -> np.ndarray:
        """Generate embeddings for a list of concept texts.

        Args:
            concept_texts: List of formatted concept descriptions.
            batch_size: Batch size for inference (default: self.batch_size).
            checkpoint_path: Path to save intermediate embeddings (optional).
            checkpoint_interval: Number of concepts between checkpoints.

        Returns:
            2D numpy array of embeddings (n_concepts x embedding_dim).

        """
        if not concept_texts:
            return np.array([])

        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        all_embeddings, processed_count = self._load_checkpoint(
            checkpoint_path,
            len(concept_texts),
        )

        for i in tqdm(
            range(processed_count, len(concept_texts), effective_batch_size),
            desc="Generating embeddings",
        ):
            batch_texts = concept_texts[i : i + effective_batch_size]

            if self._model_type == "hf":
                batch_embeddings_list = self._embed_batch_hf(batch_texts)

            elif self._model_type == "transformers":
                batch_embeddings_list = self._embed_batch_transformers(batch_texts)

            elif self._model_type == "ollama":
                batch_embeddings_list = self._embed_batch_ollama(batch_texts)
            else:
                msg = f"Unknown model type: {self._model_type}"
                raise ValueError(msg)

            all_embeddings.extend(batch_embeddings_list)

            new_processed_count = processed_count + len(batch_texts)
            if (
                checkpoint_path
                and new_processed_count // checkpoint_interval
                > (processed_count - 1) // checkpoint_interval
            ):
                temp_data = {"embeddings": list(all_embeddings)}
                with pathlib.Path(checkpoint_path).open("wb") as f:
                    pickle.dump(temp_data, f)  # Trusted pickle - internal files only
                logger.info("Checkpoint saved at %d concepts", new_processed_count)
            processed_count = new_processed_count

        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)
        else:
            msg = "No embeddings were generated"
            raise ValueError(msg)

        return embeddings_array

    def export_embeddings(
        self,
        embeddings_dict: dict[str, np.ndarray],
        output_path: str,
    ) -> None:
        """Save embeddings to disk.

        Args:
            embeddings_dict: Dictionary mapping cui -> embedding vector.
            output_path: Path to save embeddings (.pkl or .npz).

        """
        output_dir = pathlib.Path(output_path).parent
        if str(output_dir) and str(output_dir) != ".":
            pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

        if output_path.endswith(".pkl"):
            with pathlib.Path(output_path).open("wb") as f:
                pickle.dump(embeddings_dict, f)  # Trusted pickle - internal files only
        elif output_path.endswith(".npz"):
            np.savez(output_path, **embeddings_dict)
        else:
            with pathlib.Path(output_path).open("wb") as f:
                pickle.dump(embeddings_dict, f)  # Trusted pickle - internal files only

    def export_dataframe(self, df: pd.DataFrame, output_path: str) -> None:
        """Export concept DataFrame to CSV/Parquet."""
        output_dir = pathlib.Path(output_path).parent
        if str(output_dir) and str(output_dir) != ".":
            pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)


class ConceptVectorSearch:
    """FAISS-based vector similarity search for clinical concepts."""

    @staticmethod
    def load_embeddings_from_file(path: str) -> dict[str, np.ndarray]:
        """Load embeddings from disk.

        Args:
            path: Path to .pkl or .npz file containing embeddings

        Returns:
            Dictionary mapping cui -> embedding array

        """
        if path.endswith(".pkl"):
            with pathlib.Path(path).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
        elif path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        else:
            with pathlib.Path(path).open("rb") as f:
                data = pickle.load(f)  # noqa: S301

        # Handle both old format (just dict) and new format with
        # embeddings and names keys
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        if isinstance(data, dict):
            return data
        msg = f"Unknown embeddings file format: {path}"
        raise ValueError(msg)

    def __init__(
        self,
        embeddings_dict_or_path: str | dict[str, np.ndarray] | dict[str, any],
        cui_to_name: dict[str, str] | None = None,
        embedder: ClinicalConceptEmbedder = None,
    ) -> None:
        """Initialize search with pre-computed embeddings.

        Args:
            embeddings_dict_or_path: Either a dict of {cui: embedding} or path
                           to .pkl/.npz file containing embeddings.
                           Can also be a dict with 'embeddings' and 'names' keys.
            cui_to_name: Optional mapping from CUI to concept name for search results.
            embedder: Optional ClinicalConceptEmbedder instance. If provided, its
                     model will be reused for query embedding.

        """
        self.embeddings_dict = None
        self.cui_to_name = cui_to_name or {}
        self.embedder = embedder
        self.index = None
        self.cui_list = None
        self.embedding_dim = None

        if isinstance(embeddings_dict_or_path, str):
            self._load_embeddings_from_file(embeddings_dict_or_path)
        elif isinstance(embeddings_dict_or_path, dict):
            if "embeddings" in embeddings_dict_or_path:
                self.embeddings_dict = embeddings_dict_or_path["embeddings"]
                if "names" in embeddings_dict_or_path:
                    self.cui_to_name = embeddings_dict_or_path["names"]
            else:
                self.embeddings_dict = embeddings_dict_or_path

    def _load_embeddings_from_file(self, path: str) -> None:
        """Load embeddings from disk."""
        if path.endswith(".pkl"):
            with pathlib.Path(path).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
            # Handle both old format (just dict) and new format (with names)
            if isinstance(data, dict) and "embeddings" in data:
                self.embeddings_dict = data["embeddings"]
                if "names" in data:
                    self.cui_to_name = data["names"]
            else:
                self.embeddings_dict = data
        elif path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            self.embeddings_dict = {k: data[k] for k in data.files}
        else:
            with pathlib.Path(path).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
            if isinstance(data, dict) and "embeddings" in data:
                self.embeddings_dict = data["embeddings"]
                if "names" in data:
                    self.cui_to_name = data["names"]
            else:
                self.embeddings_dict = data

    def build_index(self, index_type: str = "FlatIP") -> None:
        """Build FAISS index from embeddings.

        Args:
            index_type: FAISS index type ('FlatIP' for Inner Product / Cosine
                       after L2 normalization, 'HNSW' for faster approximate search).

        """
        if faiss is None:
            msg = (
                "FAISS not installed. "
                "Run: pip install faiss-cpu or pip install faiss-gpu"
            )
            raise RuntimeError(
                msg,
            )

        if not self.embeddings_dict:
            msg = "No embeddings loaded"
            raise ValueError(msg)

        cuis = list(self.embeddings_dict.keys())
        embeddings = np.array([self.embeddings_dict[cui] for cui in cuis])

        if len(embeddings) == 0:
            msg = "Empty embeddings"
            raise ValueError(msg)

        self.embedding_dim = embeddings.shape[1]
        self.cui_list = cuis

        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            msg = f"Unknown index type: {index_type}"
            raise ValueError(msg)

        self.index.add(embeddings_normalized.astype(np.float32))

    def search(
        self,
        query_text: str | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, str, float]]:
        """Search for similar concepts.

        Args:
            query_text: Query concept description.
                Optional if query_embedding provided.
            query_embedding: Pre-computed query embedding array.
            top_k: Number of results to return.

        Returns:
            List of (cui, concept_name, similarity_score) tuples.

        """
        if self.index is None:
            msg = "Index not built. Call build_index() first"
            raise ValueError(msg)

        if query_embedding is None:
            query_embedding = self._get_query_embedding(query_text)

        if query_embedding is None:
            return []

        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(
            query_normalized.reshape(1, -1).astype(np.float32),
            top_k,
        )

        results = []
        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.cui_list):
                cui = self.cui_list[idx]
                score = float(distances[0][i])
                concept_name = self.cui_to_name.get(cui)
                if concept_name is None:
                    emb_or_name = self.embeddings_dict.get(cui, "Unknown Concept")
                    if isinstance(emb_or_name, str):
                        concept_name = emb_or_name
                    else:
                        concept_name = f"CUI: {cui}"
                results.append((cui, concept_name, score))

        return results

    def _get_query_embedding(
        self,
        query_text: str | None = None,
    ) -> np.ndarray | None:
        """Get embedding for a query string using the configured embedder model.

        Args:
            query_text: Query text to embed.

        Returns:
            Embedding vector or None if failed.

        """
        if self.embedder is not None:
            try:
                texts = [query_text] if isinstance(query_text, str) else query_text
                return self.embedder.generate_embeddings(texts, batch_size=1)[0]
            except (KeyError, TypeError, AttributeError) as e:
                msg = f"Failed to embed query using configured embedder: {e}"
                raise RuntimeError(
                    msg,
                ) from None

        try:
            texts = [query_text] if isinstance(query_text, str) else query_text
            return self.embedder.generate_embeddings(texts, batch_size=1)[0]
        except (KeyError, TypeError, AttributeError) as e:
            msg = f"Failed to embed query using configured embedder: {e}"
            raise RuntimeError(
                msg,
            ) from None

        if SentenceTransformer is None:
            msg = "sentence-transformers not installed"
            raise RuntimeError(msg)
        return None


class _MedCatCAT:
    """Type stub for MedCAT CAT."""

    cdb: object


def load_concepts_from_medcat(cat: _MedCatCAT) -> pd.DataFrame:
    """Load concepts from a MedCAT CAT object.

    Args:
        cat: MedCAT CAT instance with loaded model pack.

    Returns:
        DataFrame with concept information (cui, preferred_name, synonyms, type_id).

    """
    cdb = cat.cdb

    concepts_data = []

    for cui in tqdm(cdb.cui2preferred_name.keys(), desc="Loading MedCAT concepts"):
        pref_name = cdb.cui2preferred_name.get(cui, "")

        synonyms_list = []
        if cui in cdb.cui2names:
            syns = cdb.cui2names.get(cui, [])
            if isinstance(syns, dict):
                synonyms_list.extend(list(syns.keys()))
            elif isinstance(syns, list):
                synonyms_list.extend(syns)

        type_ids = []
        if cui in cdb.cui2type_ids:
            type_ids = list(cdb.cui2type_ids.get(cui, []))

        concepts_data.append(
            {
                "cui": str(cui),
                "preferred_name": pref_name,
                "synonyms": "; ".join(str(s) for s in synonyms_list[:10]),
                "type_id": "; ".join(str(t) for t in type_ids[:5]),
            },
        )

    return pd.DataFrame(concepts_data)


class _MedCatCDB:
    """Type stub for MedCAT CDB."""

    cui2preferred_name: dict


def load_concepts_from_cdb(cdb: _MedCatCDB) -> pd.DataFrame:
    """Load concepts from a MedCAT ConceptDatabase directly.

    Args:
        cdb: MedCAT ConceptDatabase instance.

    Returns:
        DataFrame with concept information (cui, preferred_name, synonyms, type_id).

    """
    concepts_data = []

    for cui in tqdm(cdb.cui2preferred_name.keys(), desc="Loading CDB concepts"):
        pref_name = cdb.cui2preferred_name.get(cui, "")

        synonyms_list = []
        if cui in cdb.cui2names:
            syns = cdb.cui2names.get(cui, [])
            if isinstance(syns, dict):
                synonyms_list.extend(list(syns.keys()))
            elif isinstance(syns, list):
                synonyms_list.extend(syns)

        type_ids = []
        if cui in cdb.cui2type_ids:
            type_ids = list(cdb.cui2type_ids.get(cui, []))

        concepts_data.append(
            {
                "cui": str(cui),
                "preferred_name": pref_name,
                "synonyms": "; ".join(str(s) for s in synonyms_list[:10]),
                "type_id": "; ".join(str(t) for t in type_ids[:5]),
            },
        )

    return pd.DataFrame(concepts_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LLM embeddings for SNOMED/ MedCAT concepts",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "ollama"],
        default="hf",
        help="Embedding backend (default: hf)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name or path (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--medcat-path",
        type=str,
        default=None,
        help="Path to MedCAT model pack (.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--concept-term",
        type=str,
        default="meningioma",
        help="Term to search and expand (default: meningioma)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference",
    )

    args = parser.parse_args()

    logger.info("Starting clinical concept embedding pipeline...")
    logger.info("Backend: %s", args.backend)
    logger.info("Model: %s", args.model)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if args.backend == "hf":
        embedder = ClinicalConceptEmbedder(
            model_name_or_path=args.model,
            backend="hf",
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        embedder = ClinicalConceptEmbedder(
            model_name_or_path=args.model,
            backend="ollama",
            device=args.device,
            batch_size=args.batch_size,
        )

    if args.medcat_path:
        from medcat.cat import CAT

        logger.info("Loading MedCAT model pack: %s", args.medcat_path)
        cat = CAT.load_model_pack(args.medcat_path)
        concept_df = load_concepts_from_medcat(cat)
    else:
        from snomed_term_lookup import create_term_lookup_from_directory

        default_snomed_dir = (
            pathlib.Path(__file__).parent.parent
            / "uk_sct2cl_42.2.0"
            / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z"
        )
        snomed_dir = os.environ.get("SNOMED_DIR", default_snomed_dir)

        logger.info("Loading SNOMED concepts from: %s", snomed_dir)
        lookup = create_term_lookup_from_directory(snomed_dir)
        results = lookup.find_concepts_by_term(
            args.concept_term,
            match_prefix=True,
            top_n=100,
        )

        concept_df = pd.DataFrame([{"cui": c, "preferred_name": t} for c, t in results])

    logger.info("Prepared %d concepts", len(concept_df))

    concept_texts = embedder.prepare_concept_text(concept_df)

    embeddings = embedder.generate_embeddings(concept_texts, batch_size=args.batch_size)

    cui_to_embedding = {}
    for i, cui in enumerate(concept_df["cui"].tolist()):
        cui_to_embedding[cui] = embeddings[i]

    output_path = pathlib.Path(args.output_dir) / "concept_embeddings.pkl"
    embedder.export_embeddings(cui_to_embedding, output_path)
    logger.info("Saved embeddings to: %s", output_path)

    search_engine = ConceptVectorSearch(
        cui_to_embedding,
        cui_to_name={
            cui: concept_df[concept_df["cui"] == cui]["preferred_name"].iloc[0]
            for cui in concept_df["cui"]
        },
        embedder=embedder,
    )
    search_engine.build_index(index_type="FlatIP")

    logger.info("\nQuerying for similar concepts to '%s'...", args.concept_term)

    query_results = search_engine.search(args.concept_term, top_k=20)

    logger.info("\nTop 20 similar concepts:")
    for i, (cui, name, score) in enumerate(query_results[:20], 1):
        logger.info("%2d. CUI: %15s | Score: %.4f", i, cui, score)
        logger.info("     Name: %s", name)
