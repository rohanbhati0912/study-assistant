from unittest.mock import MagicMock

from agent_handler import StudyAssistantHandler


def make_handler(relevant_docs):
    handler = StudyAssistantHandler.__new__(StudyAssistantHandler)
    handler.rag_helper = MagicMock()
    handler.rag_helper.query_with_sources.return_value = relevant_docs
    handler.config = {"prompts": {"rag_query": {"base": "{question}\n{context}"}}}
    handler.agents = MagicMock()
    handler._run_agent = MagicMock(return_value="Tutor answer")
    return handler


def test_query_documents_includes_source_and_one_based_page_citations():
    handler = make_handler(
        [
            {
                "content": "Photosynthesis occurs in chloroplasts.",
                "source": "biology_notes.pdf",
                "page": 3,
            }
        ]
    )

    result = handler.query_documents("Where does photosynthesis occur?")

    assert result == "Tutor answer"
    handler._run_agent.assert_called_once()
    prompt = handler._run_agent.call_args.args[1]
    assert "[Source: biology_notes.pdf; Page: 4]" in prompt
    assert "Photosynthesis occurs in chloroplasts." in prompt


def test_query_documents_handles_missing_source_metadata():
    handler = make_handler([{"content": "Example passage.", "source": None, "page": None}])

    result = handler.query_documents("Summarize this passage.")

    assert result == "Tutor answer"
    prompt = handler._run_agent.call_args.args[1]
    assert "[Retrieved passage 1; source metadata unavailable]" in prompt
    assert "Example passage." in prompt
