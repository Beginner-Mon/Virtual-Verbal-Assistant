"""A0 path traversal tests — persona_id validation defense in depth."""

import pytest


@pytest.mark.unit
class TestPersonaPathTraversal:
    """A0: persona_id must not allow path traversal."""

    def test_fallback_on_traversal_id(self):
        """../../../README → fallback persona, not file read."""
        from langgraph_agents.nodes._persona_loader import get_persona
        persona = get_persona("../../../README")
        # Should return fallback (title = "ECA Default"), not crash or read external file
        assert persona["persona_id"] == "../../../README"
        assert persona["title"] == "ECA Default"

    def test_fallback_on_path_separator(self):
        """a/b, a\\b → fallback persona."""
        from langgraph_agents.nodes._persona_loader import get_persona
        for bad_id in ["a/b", "a\\b", "..%2F..%2Fx"]:
            persona = get_persona(bad_id)
            assert persona["title"] == "ECA Default", f"FAIL for {bad_id}"

    def test_valid_personas_still_load(self):
        """Valid persona IDs must still work."""
        from langgraph_agents.nodes._persona_loader import get_persona
        for good_id in ["eca_default", "eca_clinical", "eca_friendly"]:
            persona = get_persona(good_id)
            assert persona["persona_id"] == good_id

    def test_cache_not_polluted_by_invalid(self):
        """Invalid persona must not be cached as valid."""
        from langgraph_agents.nodes._persona_loader import get_persona, _persona_cache

        # Ensure cache is loaded with a valid persona first
        get_persona("eca_default")
        cached_default = _persona_cache.get("eca_default")
        assert cached_default is not None
        assert cached_default["persona_id"] == "eca_default"

        # Load invalid — should get fallback
        invalid = get_persona("../../etc/passwd")
        assert invalid["title"] == "ECA Default"

        # The valid entry must NOT be corrupted by the invalid lookup
        assert _persona_cache["eca_default"] is cached_default

        # Invalid id returns fallback but is NOT cached (no unbounded growth from bad ids)
        assert "../../etc/passwd" not in _persona_cache

        # Repeated invalid lookup stays a fallback (never reads an external file)
        assert get_persona("../../etc/passwd")["title"] == "ECA Default"
