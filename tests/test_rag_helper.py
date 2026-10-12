from unittest.mock import MagicMock

from rag_helper import RAGHelper


def make_helper():
    """Create a helper without initializing embeddings or a database."""
    helper = RAGHelper.__new__(RAGHelper)
    helper.vectorstore = None
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


def test_query_with_sources_returns_content_and_metadata():
    helper = make_helper()
    helper.vectorstore = MagicMock()

    document = MagicMock()
    document.page_content = "Python is a programming language."
    document.metadata = {"source": "python_notes.pdf", "page": 1}
    helper.vectorstore.similarity_search.return_value = [document]

    results = helper.query_with_sources("What is Python?")

    assert results == [
        {
            "content": "Python is a programming language.",
            "source": "python_notes.pdf",
            "page": 1,
        }
    ]


def test_query_with_sources_handles_missing_metadata():
    helper = make_helper()
    helper.vectorstore = MagicMock()

    document = MagicMock()
    document.page_content = "Example study notes."
    document.metadata = {}
    helper.vectorstore.similarity_search.return_value = [document]

    results = helper.query_with_sources("Summarize my notes")

    assert results == [
        {
            "content": "Example study notes.",
            "source": None,
            "page": None,
        }
    ]


def test_query_with_sources_returns_empty_without_vectorstore():
    helper = make_helper()

    assert helper.query_with_sources("What is Python?") == []


def test_load_pdf_preserves_page_and_uses_original_source_name(monkeypatch):
    helper = make_helper()
    helper.text_splitter = MagicMock()
    helper.text_splitter.split_documents.side_effect = lambda docs: docs
    helper.vectorstore = MagicMock()

    document = MagicMock()
    document.metadata = {"source": "/tmp/upload-123.pdf", "page": 3}
    loader = MagicMock()
    loader.load.return_value = [document]
    monkeypatch.setattr("rag_helper.PyPDFLoader", lambda _: loader)

    assert helper.load_pdf("/tmp/upload-123.pdf", source_name="biology_notes.pdf")
    assert document.metadata == {"source": "biology_notes.pdf", "page": 3}
    helper.vectorstore.add_documents.assert_called_once_with([document])


def test_load_text_uses_original_source_name(monkeypatch):
    helper = make_helper()
    helper.text_splitter = MagicMock()
    helper.text_splitter.split_documents.side_effect = lambda docs: docs
    helper.vectorstore = MagicMock()

    document = MagicMock()
    document.metadata = {"source": "/tmp/upload-123.txt"}
    loader = MagicMock()
    loader.load.return_value = [document]
    monkeypatch.setattr("rag_helper.TextLoader", lambda _: loader)

    assert helper.load_text("/tmp/upload-123.txt", source_name="biology_notes.txt")
    assert document.metadata == {"source": "biology_notes.txt"}
    helper.vectorstore.add_documents.assert_called_once_with([document])
