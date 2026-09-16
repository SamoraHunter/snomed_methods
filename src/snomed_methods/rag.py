#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
Retrieval-Augmented Generation (RAG) Module for SNOMED CT

The RAG module provides semantic search and concept retrieval for SNOMED CT,
using embeddings to find related concepts and LLM explanations for reasoning.

Core Features:
- Embeds SNOMED concept descriptions in FAISS index using SentenceTransformer
- Retrieves relevant codes based on semantic similarity (cosine distance)
- Generates LLM-powered explanations explaining WHY codes were retrieved
- Supports multi-turn refinement queries with conversation context

Architecture:
    RAGRetriever:     FAISS vector search for concept embeddings
    RAGExplanations:  LLM explanation generation for retrievals
    RAGChat:          Multi-turn chat interface with context persistence

Example Usage:
    >>> from snomed_methods.rag import RAGRetriever, RAGExplanations, RAGChat
    >>> retriever = RAGRetriever(embedder=embedder, cui_to_embedding=cui_map)
    >>> results = retriever.retrieve("meningioma", top_k=10)
    >>> chat = RAGChat(retriever=retriever)
    >>> response = chat.ask(query="brain tumor")

Reference:
    - FAISS: https://github.com/facebookresearch/faiss
    - Sentence-Transformers: https://www.sbert.net/
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_concepts_from_cdb(cdb: Any) -> tuple:
    """Load concepts from MedCAT CDB."""
    cui_to_name = {}
    for cui, name in cdb.cui2preferred_name.items():
        if name:
            cui_to_name[cui] = str(name)
    return cui_to_name


class RAGRetriever:
    """
    RAG retriever combining FAISS vector search with LLM explanations.

    This class provides semantic concept retrieval for SNOMED CT by:
    1. Building a FAISS index from concept embeddings
    2. Embedding user queries using the same embedding model
    3. Finding similar concepts via cosine similarity in embedding space

    The retriever supports both loading pre-built indices and building new ones.

    Example:
        >>> embedder = ClinicalConceptEmbedder(model_name="all-MiniLM-L6-v2")
        >>> retriever = RAGRetriever(embedder=embedder, cui_to_embedding=cui_map)
        >>> results = retriever.retrieve("brain tumor", top_k=5)

    Attributes:
        embedder: ClinicalConceptEmbedder instance for query embedding
        index: FAISS index (IndexFlatIP or loaded from disk)
        cui_list: List of concept CUIs in the index
        cui_to_name: Mapping from CUI to concept name
    """

    def __init__(
        self,
        embedder: Any = None,
        cui_to_embedding: Optional[Dict[str, np.ndarray]] = None,
        cui_to_name: Optional[Dict[str, str]] = None,
        index_path: Optional[str] = None,
        uk_path: Optional[str] = None,
    ) -> None:
        """
        Initialize RAG retriever.

        Args:
            embedder: ClinicalConceptEmbedder instance for query embedding.
                Required if not loading from saved index.
            cui_to_embedding: Dict mapping CUI to embedding vector (optional).
                Used to build the FAISS index. Can be provided directly or
                loaded from a pre-built index file.
            cui_to_name: Dict mapping CUI to concept name (optional).
                Provides human-readable names for retrieved concepts.
            index_path: Path to saved FAISS index (.pkl or .faiss).
                If provided and exists, loads the pre-built index instead of
                building from embeddings. Overrides cui_to_embedding.
            uk_path: Path to UK Clinical RF2 directory for term lookups.
                Reserved for future integration with SNOMED lookup tools.

        Raises:
            ValueError: If no embeddings available and no index path provided.
            RuntimeError: If FAISS is not installed when needed.
        """
        self.embedder = embedder
        self.cui_to_embedding = cui_to_embedding or {}
        self.cui_to_name = cui_to_name or {}
        self.index = None
        self.embedding_dim = None

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            self._build_index()

    def _build_index(self) -> None:
        """
        Build FAISS index from loaded embeddings.

        Normalizes embeddings to unit vectors for cosine similarity computation
        using inner product (FAISS IndexFlatIP).

        Raises:
            ValueError: If no embeddings are available to build the index.
            RuntimeError: If FAISS is not installed.
        """
        try:
            import faiss
        except ImportError as e:
            msg = (
                "FAISS not installed. "
                "Run: pip install faiss-cpu or pip install faiss-gpu"
            )
            raise RuntimeError(msg) from e

        if not self.cui_to_embedding:
            raise ValueError("No embeddings available to build index")

        cuis = list(self.cui_to_embedding.keys())
        embeddings = np.array([self.cui_to_embedding[cui] for cui in cuis])

        self.embedding_dim = embeddings.shape[1]
        self.cui_list = cuis

        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_normalized.astype(np.float32))

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        return_scores: bool = True,
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve relevant concepts for a query using embedding similarity.

        Embeds the query text, then searches the FAISS index to find
        the most similar concept embeddings based on cosine distance.

        Args:
            query: Input query text (e.g., "brain tumor", "meningioma")
            top_k: Number of results to retrieve. Default is 20.
            return_scores: Whether to return similarity scores.
                If False, returns score of 1.0 for all results.

        Returns:
            List of tuples (cui, concept_name, score) sorted by similarity
            in descending order. Score is cosine similarity (range: -1 to 1).

        Raises:
            ValueError: If index has not been built or loaded.
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")

        query_embedding = self._embed_query(query)
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(
            query_normalized.reshape(1, -1).astype(np.float32), top_k
        )

        results = []
        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.cui_list):
                cui = self.cui_list[idx]
                score = float(distances[0][i])
                concept_name = self.cui_to_name.get(cui, f"CUI: {cui}")
                results.append((cui, concept_name, score if return_scores else 1.0))

        return results

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string using the configured embedder.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        return self.embedder.generate_embeddings([query], batch_size=1)[0]

    def save_index(self, path: str) -> None:
        """
        Save FAISS index to disk for later reuse.

        Persists the index, concept list, and embedding dimension
        so they don't need to be rebuilt. Useful for production deployments
        where embeddings are expensive to generate.

        Args:
            path: Path to save index (.faiss or .pkl extension).
                If extension is omitted, appends '.faiss'.

        Raises:
            RuntimeError: If FAISS is not installed.
        """
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError("FAISS not installed") from e

        import pickle

        if path.endswith(".faiss"):
            faiss.write_index(self.index, path)
        elif path.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "index": self.index,
                        "cui_list": self.cui_list,
                        "embedding_dim": self.embedding_dim,
                    },
                    f,
                )
        else:
            faiss.write_index(self.index, path + ".faiss")

    def load_index(self, path: str) -> None:
        """
        Load FAISS index from disk.

        Restores a previously saved index to avoid rebuilding.
        The loaded index must be compatible with the current embedder's
        embedding dimension.

        Args:
            path: Path to saved index (.faiss or .pkl).
                If extension is omitted, looks for '.faiss'.

        Raises:
            RuntimeError: If FAISS is not installed.
            ValueError: If the loaded index is incompatible.
        """
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError("FAISS not installed") from e

        import pickle

        if path.endswith(".faiss"):
            self.index = faiss.read_index(path)
            self.embedding_dim = self.index.d
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.index = data.get("index")
            self.cui_list = data.get("cui_list", [])
            self.embedding_dim = data.get("embedding_dim")
        else:
            self.index = faiss.read_index(path + ".faiss")
            self.embedding_dim = self.index.d


class RAGExplanations:
    """
    Generates LLM-powered explanations for retrieved SNOMED concepts.

    This class uses a language model to generate natural-language explanations
    explaining why each concept was retrieved for a given query.
    Explanations help users understand the clinical relevance of search results.

    Supported backends:
        - 'ollama': Uses local Ollama instance via API
        - 'hf': Uses Hugging Face Transformers pipeline

    Example:
        >>> embedder = ClinicalConceptEmbedder(model_name="all-MiniLM-L6-v2")
        >>> explanations = RAGExplanations(embedder=embedder, backend="ollama")
        >>> explanation = explanations.generate_explanation(
        ...     query="meningioma",
        ...     cui="C0023956",
        ...     concept_name="Meningioma",
        ...     retrieved_concepts=[("C0023956", "Meningioma", 0.85), ...]
        ... )
    """

    def __init__(
        self,
        embedder: Any = None,
        backend: str = "ollama",
        model_name: str = "qwen2.5-coder",
    ) -> None:
        """Initialize RAG explanations generator.

        Args:
            embedder: ClinicalConceptEmbedder for concept embeddings
            backend: LLM backend ("ollama" or "hf") to use for generation
            model_name: Model identifier for explanation generation:
                - Ollama models (e.g., 'qwen2.5-coder', 'llama3')
                - Hugging Face models (e.g., 'meta-llama/Llama-3-8B-Instruct')

        Note:
            For 'ollama' backend, ensure the model is available via
            `ollama pull <model_name>` command.
            For 'hf' backend, ensure transformers is installed and model is accessible.
        """
        self.embedder = embedder
        self.backend = backend
        self.model_name = model_name

    def generate_explanation(
        self,
        query: str,
        cui: str,
        concept_name: str,
        retrieved_concepts: List[Tuple[str, str, float]],
    ) -> str:
        """Generate explanation for why a concept was retrieved.

        Args:
            query: Original search query
            cui: Retrieved concept CUI
            concept_name: Retrieved concept name
            retrieved_concepts: List of all retrieved (cui, name, score) tuples

        Returns:
            Natural language explanation
        """
        top_scores = [s for _, _, s in retrieved_concepts[:5]]
        avg_score = np.mean(top_scores) if top_scores else 0

        context_items = []
        for i, (c, n, s) in enumerate(retrieved_concepts[:3]):
            context_items.append(f"  {i + 1}. [{s:.3f}] {n} ({c})")

        explanation_prompt = f"""Query: "{query}"

Retrieved Concept:
- Name: {concept_name}
- CUI: {cui}
- Retrieval Score: {top_scores[0] if top_scores else 0:.3f}

Top Retrieval Scores (for context):
{chr(10).join(context_items)}

Average Score of Top Concepts: {avg_score:.3f}

Based on the query and retrieval scores, explain why this concept was retrieved.
Focus on semantic similarity, relevant keywords, or conceptual relationships.

Provide a concise explanation (2-3 sentences) that would help a clinician
understand the relevance of this SNOMED concept to their search query."""

        if self.backend == "ollama":
            return self._generate_ollama(explanation_prompt)
        return self._generate_hf(explanation_prompt)

    def _generate_ollama(self, prompt: str) -> str:
        """Generate explanation using Ollama."""
        import ollama

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant explaining clinical concept "
                            "retrieval results. Be concise and clinically relevant."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Could not generate explanation: {e}"

    def _generate_hf(self, prompt: str) -> str:
        """Generate explanation using Hugging Face model."""
        from transformers import pipeline

        try:
            generator = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=256,
                temperature=0.3,
            )
            result = generator(prompt)[0]
            return result["generated_text"].strip()
        except Exception as e:
            return f"Could not generate explanation: {e}"

    def generate_batch_explanations(
        self,
        query: str,
        retrieved_concepts: List[Tuple[str, str, float]],
    ) -> List[Dict]:
        """Generate explanations for multiple concepts.

        Args:
            query: Original search query
            retrieved_concepts: List of (cui, name, score) tuples

        Returns:
            List of dicts with cui, name, score, explanation
        """
        results = []
        for cui, concept_name, score in retrieved_concepts[:10]:
            explanation = self.generate_explanation(
                query=query,
                cui=cui,
                concept_name=concept_name,
                retrieved_concepts=retrieved_concepts,
            )
            results.append(
                {
                    "cui": cui,
                    "name": concept_name,
                    "score": score,
                    "explanation": explanation,
                }
            )
        return results


class RAGChat:
    """Multi-turn RAG chat for SNOMED concept search with refinement."""

    def __init__(
        self,
        retriever: RAGRetriever,
        explanations: Optional[RAGExplanations] = None,
        max_context_turns: int = 5,
    ):
        """Initialize RAG chat.

        Args:
            retriever: RAGRetriever instance for concept retrieval
            explanations: Optional RAGExplanations for generating explanations
            max_context_turns: Number of previous turns to retain in context
        """
        self.retriever = retriever
        self.explanations = explanations
        self.max_context_turns = max_context_turns
        self.conversation_history: List[Dict] = []
        self.system_prompt = (
            "You are a clinical terminology assistant helping users "
            "find and understand SNOMED CT concepts. Use RAG to provide "
            "accurate, clinically relevant information."
        )

    def ask(
        self,
        query: str,
        top_k: int = 15,
        return_explanations: bool = True,
    ) -> Dict:
        """Process a user query and return results.

        Args:
            query: User input query
            top_k: Number of concepts to retrieve
            return_explanations: Whether to generate explanations

        Returns:
            Response dict with results, explanations (optional), and context
        """
        self.conversation_history.append({"role": "user", "content": query})

        retrieved = self.retriever.retrieve(query, top_k=top_k)

        response = {
            "query": query,
            "results": [{"cui": c, "name": n, "score": s} for c, n, s in retrieved],
            "context_turns": len(self.conversation_history),
        }

        if return_explanations and self.explanations:
            response["explanations"] = self.explanations.generate_batch_explanations(
                query=query,
                retrieved_concepts=retrieved,
            )

        if len(self.conversation_history) > self.max_context_turns:
            self.conversation_history = self.conversation_history[
                -self.max_context_turns :
            ]

        return response

    def follow_up(
        self,
        feedback: str,
        refine_with_previous: bool = True,
    ) -> Dict:
        """Process a follow-up query or feedback.

        Args:
            feedback: User feedback or refinement query
            refine_with_previous: Whether to incorporate previous context

        Returns:
            Response dict with refined results
        """
        if refine_with_previous and self.conversation_history:
            context_text = " | ".join(
                [m["content"] for m in self.conversation_history[-3:]]
            )
            query = f"Previous: {context_text}. Current: {feedback}"
        else:
            query = feedback

        return self.ask(query=query)

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
