"""tests/unit/test_security.py

Unit tests for security utility functions.
"""

import pytest
from app.core.security import compute_document_id, sanitise_filename


class TestComputeDocumentId:
    def test_deterministic(self):
        """Same content + filename → same ID."""
        a = compute_document_id(b"hello world", "doc.pdf")
        b = compute_document_id(b"hello world", "doc.pdf")
        assert a == b

    def test_different_content_different_id(self):
        a = compute_document_id(b"content a", "doc.pdf")
        b = compute_document_id(b"content b", "doc.pdf")
        assert a != b

    def test_different_filename_different_id(self):
        a = compute_document_id(b"same content", "doc_a.pdf")
        b = compute_document_id(b"same content", "doc_b.pdf")
        assert a != b

    def test_returns_hex_string(self):
        doc_id = compute_document_id(b"test", "test.pdf")
        assert isinstance(doc_id, str)
        assert len(doc_id) == 64  # SHA-256 hex = 64 chars


class TestSanitiseFilename:
    def test_path_traversal_removed(self):
        result = sanitise_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_normal_filename_unchanged(self):
        result = sanitise_filename("leave_policy.pdf")
        assert result == "leave_policy.pdf"

    def test_spaces_replaced(self):
        result = sanitise_filename("my document.pdf")
        assert " " not in result

    def test_empty_filename_fallback(self):
        result = sanitise_filename("")
        assert result == "uploaded_file"

    def test_special_chars_removed(self):
        result = sanitise_filename("file<>name.pdf")
        assert "<" not in result
        assert ">" not in result
