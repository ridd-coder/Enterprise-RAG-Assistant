"""tests/unit/test_generator.py

Unit tests for LLM generation, hallucination guard, and prompt construction.
"""

from unittest.mock import MagicMock, patch

from app.rag.generator import NO_CONTEXT_ANSWER, LLMGenerator
from app.rag.prompt import build_context_block, build_messages
from app.rag.vector_store import SearchResult


def make_search_result(score: float, doc_id: str = "doc1", page: int = 1, text: str = "sample"):
    payload = {
        "document_id": doc_id,
        "filename": f"{doc_id}.pdf",
        "page_number": page,
        "chunk_index": 0,
        "text": text,
    }
    return SearchResult(payload=payload, score=score, point_id="pt1")


class TestPromptBuilder:
    def test_build_context_block_empty(self):
        block = build_context_block([])
        assert "No relevant context" in block

    def test_build_context_block_formatting(self):
        chunks = [
            make_search_result(score=0.9, doc_id="policy", page=3, text="Annual leave is 20 days.")
        ]
        block = build_context_block(chunks)
        assert "[1] Source: policy.pdf, Page 3" in block
        assert "Annual leave is 20 days." in block

    def test_build_messages_structure(self):
        chunks = [make_search_result(score=0.9, doc_id="handbook", page=1, text="Handbook content")]
        messages = build_messages("How many days?", chunks)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "How many days?" in messages[1]["content"]


class TestLLMGenerator:
    @patch("app.rag.generator.get_settings")
    @patch("app.rag.generator.OpenAI")
    def test_hallucination_guard_no_context(self, mock_openai, mock_settings):
        """When no chunks are provided, generator MUST return sentinel without calling OpenAI."""
        mock_settings.return_value.require_openai_key.return_value = "fake-key"
        mock_settings.return_value.openai_model = "gpt-4o-mini"
        mock_settings.return_value.openai_max_tokens = 1024
        mock_settings.return_value.openai_temperature = 0.0

        generator = LLMGenerator()
        result = generator.generate("What is the leave policy?", chunks=[])

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.tokens_used == 0
        # Crucial check: client was never called
        generator._client.chat.completions.create.assert_not_called()

    @patch("app.rag.generator.get_settings")
    @patch("app.rag.generator.OpenAI")
    def test_generation_with_chunks(self, mock_openai, mock_settings):
        mock_settings.return_value.require_openai_key.return_value = "fake-key"
        mock_settings.return_value.openai_model = "gpt-4o-mini"
        mock_settings.return_value.openai_max_tokens = 1024
        mock_settings.return_value.openai_temperature = 0.0

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Employees are entitled to 20 days. [Source: leave.pdf, Page 4]"
        )
        mock_response.usage.total_tokens = 95
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        generator = LLMGenerator()
        chunks = [make_search_result(score=0.92, doc_id="leave", page=4, text="20 days leave")]
        result = generator.generate("What is leave?", chunks=chunks)

        assert "20 days" in result.answer
        assert result.tokens_used == 95
        generator._client.chat.completions.create.assert_called_once()
