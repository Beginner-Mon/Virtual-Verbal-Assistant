import importlib.util
import pathlib
import unittest


def _load_knowledge_librarian_class():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = project_root / "agents" / "knowledge_librarian.py"
    spec = importlib.util.spec_from_file_location("isolated_knowledge_librarian", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.KnowledgeLibrarian


KnowledgeLibrarian = _load_knowledge_librarian_class()


class TestKnowledgeLibrarianEntityTags(unittest.TestCase):
    def setUp(self) -> None:
        # Bypass full __init__ to avoid external service setup.
        self.librarian = KnowledgeLibrarian.__new__(KnowledgeLibrarian)
        self.librarian._entity_spoke_enabled = False
        self.librarian._entity_slm_available = False
        self.librarian._entity_slm_client = None

    def test_extract_query_entity_tags_local_fallback(self) -> None:
        tags = self.librarian._extract_query_entity_tags(
            "I need lower back pain mobility exercises",
            source_hint="documents",
        )
        self.assertIn("back", tags)
        self.assertIn("pain_relief", tags)
        self.assertIn("mobility", tags)

    def test_apply_entity_tag_rerank_boosts_overlap(self) -> None:
        raw_results = [
            {
                "document": "neck mobility drill",
                "similarity": 0.70,
                "metadata": {"entity_tags": "neck,mobility"},
                "strategy": "vector",
            },
            {
                "document": "lower back rehab routine",
                "similarity": 0.64,
                "metadata": {"entity_tags": "back,mobility,rehab"},
                "strategy": "vector",
            },
        ]

        reranked = self.librarian._apply_entity_tag_rerank(raw_results, ["back", "mobility"])
        self.assertEqual(reranked[0]["document"], "lower back rehab routine")
        self.assertEqual(reranked[0]["entity_tag_overlap"], ["back", "mobility"])


if __name__ == "__main__":
    unittest.main()
