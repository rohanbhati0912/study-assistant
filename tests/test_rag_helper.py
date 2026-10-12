from unittest.mock import MagicMock

from rag_helper import RAGHelper


def make_helper():
    """Create a helper without initializing embeddings or a database."""
    helper = RAGHelper.__new__(RAGHelper)
    helper.vectorstore = None
    helper._processed_hashes = set()
    return helper


def test_load_text_content_returns_false_without_vectorstore():
    helper = make_helper()
    helper.text_splitter = MagicMock()

    assert helper.load_text_content("Example study notes") is False


def test_query_returns_empty_list_without_vectorstore():
    helper = make_helper()

    assert helper.query("What is Python?") == []


def test_get_document_count_returns_zero_without_vectorstore():
    helper = make_helper()

    assert helper.get_document_count() == 0


def test_load_text_content_adds_documents_when_store_exists():
    helper = make_helper()
    helper.text_splitter = MagicMock()
    helper.text_splitter.split_documents.return_value = ["chunk"]
    helper.vectorstore = MagicMock()

    assert helper.load_text_content("Example notes") is True
    helper.vectorstore.add_documents.assert_called_once_with(["chunk"])


def test_get_file_hash_matches_identical_contents(tmp_path):
    helper = make_helper()
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "renamed.txt"

    first_file.write_bytes(b"Example study notes")
    second_file.write_bytes(b"Example study notes")

    assert helper._get_file_hash(str(first_file)) == helper._get_file_hash(
        str(second_file)
    )


def test_load_text_skips_duplicate_content(tmp_path):
    helper = make_helper()
    helper.vectorstore = MagicMock()
    helper.text_splitter = MagicMock()
    helper.text_splitter.split_documents.return_value = ["chunk"]

    file_path = tmp_path / "notes.txt"
    file_path.write_text("Example study notes", encoding="utf-8")

    assert helper.load_text(str(file_path)) is True
    assert helper.load_text(str(file_path)) is True

    helper.vectorstore.add_documents.assert_called_once_with(["chunk"])
