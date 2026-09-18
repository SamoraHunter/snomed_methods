#!/usr/bin/env python3
"""Pytest unit tests for RAG (Retrieval-Augmented Generation) module."""

import os
import pathlib
import tempfile

import numpy as np
import pytest


class MockEmbedder:
    """Mock embedder for testing RAG classes."""

    def generate_embeddings(self, texts, batch_size=None):
        """Generate mock embeddings."""
        return [np.random.randn(384) for _ in texts]


@pytest.fixture(scope="module")
def mock_embedder():
    """Create mock embedder fixture."""
    return MockEmbedder()


@pytest.fixture(scope="module")
def sample_data_embeddings():
    """Create sample CUI embedding data."""
    np.random.seed(42)
    return {
        "C0023956": np.random.randn(384),
        "C0013421": np.random.randn(384),
        "C0011861": np.random.randn(384),
        "C0001395": np.random.randn(384),
    }


@pytest.fixture(scope="module")
def sample_data_names():
    """Create sample CUI to name mapping."""
    return {
        "C0023956": "Meningioma",
        "C0013421": "Diabetes Mellitus",
        "C0011861": "Hypertension",
        "C0001395": "Bronchitis",
    }


class TestRAGRetrieverInit:
    """Tests for RAGRetriever initialization."""

    def test_init_with_embedder(self, mock_embedder) -> None:
        from src.snomed_methods.rag import RAGRetriever

        cui_to_embedding = {"C001": np.random.randn(384)}
        cui_to_name = {"C001": "Test Concept"}
        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=cui_to_embedding,
            cui_to_name=cui_to_name,
        )
        assert retriever.embedder is not None
        assert len(retriever.cui_to_embedding) == 1
        assert retriever.index is not None

    def test_init_without_embedder(self) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with pytest.raises(ValueError, match="No embeddings available to build index"):
            RAGRetriever()

    def test_init_with_index_path(
        self,
        mock_embedder,
        sample_data_embeddings,
        sample_data_names,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index.pkl")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
                cui_to_name=dict(sample_data_names),
            )
            retriever1.save_index(index_path)
            retriever2 = RAGRetriever(index_path=index_path)
            assert retriever2.index is not None
            assert len(retriever2.cui_list) == 4

    def test_init_with_index_only(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index.pkl")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            retriever2 = RAGRetriever(index_path=index_path)
            assert retriever2.index is not None
            assert len(retriever2.cui_list) == 4


class TestRAGRetrieverRetrieve:
    """Tests for RAGRetriever retrieve method."""

    def test_retrieve_basic(
        self,
        mock_embedder,
        sample_data_embeddings,
        sample_data_names,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
            cui_to_name=dict(sample_data_names),
        )
        results = retriever.retrieve("brain tumor", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        if len(results) > 0:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 3

    def test_retrieve_without_index(self) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = object.__new__(RAGRetriever)
        retriever.index = None
        with pytest.raises(ValueError, match="Index not built or loaded"):
            retriever.retrieve("test query")

    def test_retrieve_return_scores_false(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        results = retriever.retrieve("test", top_k=5, return_scores=False)
        if len(results) > 0:
            assert results[0][2] == 1.0

    def test_retrieve_top_k_1(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        results = retriever.retrieve("test", top_k=1)
        assert len(results) <= 1


class TestRAGRetrieverIndexOperations:
    """Tests for index save/load operations."""

    def test_save_faiss_with_extension(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = pathlib.Path(tmpdir) / "rag_index.faiss"
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            assert os.path.exists(index_path)
            assert index_path.stat().st_size > 0

    def test_save_pickle_with_extension(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = pathlib.Path(tmpdir) / "rag_index.pkl"
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            assert os.path.exists(index_path)
            assert index_path.stat().st_size > 0

    def test_save_with_no_extension(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            assert os.path.exists(index_path + ".faiss")

    def test_load_faiss_file(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index.faiss")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            retriever2 = object.__new__(RAGRetriever)
            retriever2.load_index(index_path)
            assert retriever2.index is not None
            assert hasattr(retriever2, "embedding_dim")

    def test_load_pickle_file(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index.pkl")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            retriever2 = object.__new__(RAGRetriever)
            retriever2.load_index(index_path)
            assert retriever2.index is not None
            assert len(retriever2.cui_list) == 4

    def test_load_with_no_extension(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_index")
            retriever1 = RAGRetriever(
                embedder=mock_embedder,
                cui_to_embedding=dict(sample_data_embeddings),
            )
            retriever1.save_index(index_path)
            retriever2 = object.__new__(RAGRetriever)
            retriever2.load_index(index_path + ".faiss")
            assert retriever2.index is not None
            assert hasattr(retriever2, "embedding_dim")


class TestRAGExplanationsInit:
    """Tests for RAGExplanations initialization."""

    def test_init_default_backend(self) -> None:
        from src.snomed_methods.rag import RAGExplanations

        explanations = RAGExplanations()
        assert explanations.backend == "ollama"

    def test_init_ollama_backend(self, mock_embedder) -> None:
        from src.snomed_methods.rag import RAGExplanations

        explanations = RAGExplanations(embedder=mock_embedder, backend="ollama")
        assert explanations.backend == "ollama"

    def test_init_hf_backend(self, mock_embedder) -> None:
        from src.snomed_methods.rag import RAGExplanations

        explanations = RAGExplanations(embedder=mock_embedder, backend="hf")
        assert explanations.backend == "hf"

    def test_init_custom_model(self, mock_embedder) -> None:
        from src.snomed_methods.rag import RAGExplanations

        explanations = RAGExplanations(model_name="custom-model")
        assert explanations.model_name == "custom-model"


class TestRAGChatInit:
    """Tests for RAGChat initialization."""

    def test_init_with_retriever(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        assert chat.retriever is not None
        assert len(chat.conversation_history) == 0

    def test_init_with_explanations(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGChat, RAGExplanations, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        explanations = RAGExplanations(embedder=mock_embedder)
        chat = RAGChat(retriever=retriever, explanations=explanations)
        assert chat.explanations is not None

    def test_init_custom_max_turns(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever, max_context_turns=10)
        assert chat.max_context_turns == 10


class TestRAGChatAsk:
    """Tests for RAGChat ask method."""

    def test_ask_basic(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        response = chat.ask(query="brain tumor", top_k=3)
        assert "query" in response
        assert "results" in response
        assert len(response["results"]) <= 3

    def test_ask_with_explanations(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGExplanations, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        explanations = RAGExplanations(embedder=mock_embedder)
        chat = RAGChat(retriever=retriever, explanations=explanations)
        response = chat.ask(query="test", return_explanations=True)
        if "explanations" in response:
            assert isinstance(response["explanations"], list)

    def test_ask_stores_history(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        chat.ask(query="first question")
        chat.ask(query="second question")
        assert len(chat.conversation_history) == 2

    def test_ask_with_max_context_limit(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever, max_context_turns=3)
        for i in range(5):
            chat.ask(query=f"question {i}")
        assert len(chat.conversation_history) == 3


class TestRAGChatFollowUp:
    """Tests for RAGChat follow_up method."""

    def test_follow_up_basic(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        chat.ask(query="original query")
        response = chat.follow_up(feedback="refined")
        assert "results" in response

    def test_follow_up_without_refine(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        response = chat.follow_up(
            feedback="independent query",
            refine_with_previous=False,
        )
        assert "results" in response

    def test_follow_up_stores_history(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        chat.ask(query="original")
        chat.follow_up(feedback="refined")
        assert len(chat.conversation_history) == 2


class TestRAGChatReset:
    """Tests for RAGChat reset method."""

    def test_reset_basic(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        chat.ask(query="test question")
        chat.reset()
        assert len(chat.conversation_history) == 0

    def test_reset_empty_history(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        chat.reset()
        assert len(chat.conversation_history) == 0


class TestRAGEdgeCases:
    """Tests for edge cases in RAG module."""

    def test_retrieve_with_empty_query(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        results = retriever.retrieve("", top_k=5)
        assert isinstance(results, list)

    def test_chat_with_empty_query(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)
        response = chat.ask(query="")
        assert "query" in response
        assert response["query"] == ""

    def test_retriever_with_very_large_top_k(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        from src.snomed_methods.rag import RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        results = retriever.retrieve("test", top_k=1000)
        assert len(results) <= 4


class TestIntegration:
    """Integration tests for RAG module."""

    def test_complete_rag_workflow(self, mock_embedder, sample_data_embeddings) -> None:
        from src.snomed_methods.rag import RAGChat, RAGExplanations, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        explanations = RAGExplanations(embedder=mock_embedder)
        chat = RAGChat(retriever=retriever, explanations=explanations)
        response1 = chat.ask(query="brain related condition", top_k=5)
        assert len(response1["results"]) <= 5
        response2 = chat.follow_up(feedback="make it more specific")
        assert "results" in response2
        chat.reset()
        assert len(chat.conversation_history) == 0

    def test_conversation_history_append(
        self,
        mock_embedder,
        sample_data_embeddings,
    ) -> None:
        """Test that conversation history is properly appended."""
        from src.snomed_methods.rag import RAGChat, RAGRetriever

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=dict(sample_data_embeddings),
        )
        chat = RAGChat(retriever=retriever)

        assert len(chat.conversation_history) == 0
        chat.ask(query="test query 1")
        assert len(chat.conversation_history) == 1
        assert chat.conversation_history[0] == {
            "role": "user",
            "content": "test query 1",
        }

        chat.ask(query="test query 2")
        assert len(chat.conversation_history) == 2
        assert chat.conversation_history[1] == {
            "role": "user",
            "content": "test query 2",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
