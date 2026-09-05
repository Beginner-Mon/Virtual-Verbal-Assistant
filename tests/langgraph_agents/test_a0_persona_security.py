"""A0 path traversal tests — persona_id validation defense in depth.

The security property is unchanged: a persona_id from a request body becomes
part of a filesystem path, and no crafted id may read a file outside
`personas/`. What changed on 04-09 is the OUTCOME of a rejected id.

Before, a bad id returned a generic "ECA Default" stand-in. That was safe but
dishonest — a broken or missing character silently became a different one, and
the user kept looking at Bronya's avatar while reading words written for nobody.
Now it raises `PersonaError`. The traversal is still refused; it is refused
loudly. Owner's call: answering as the wrong character is worse than not
answering.
"""

import pytest

from langgraph_agents.nodes._persona_loader import (
    PersonaError,
    _persona_cache,
    get_persona,
)


@pytest.mark.unit
class TestPersonaPathTraversal:
    """A0: persona_id must not allow path traversal."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../../../README",
            "a/b",
            "a\\b",
            "..%2F..%2Fx",
            "../../etc/passwd",
        ],
    )
    def test_traversal_id_is_refused(self, bad_id):
        """No file outside personas/ is read, and no stand-in is returned."""
        with pytest.raises(PersonaError):
            get_persona(bad_id, "vi")

    def test_valid_personas_still_load(self):
        """The four real characters. eca_* were deleted on 04-09."""
        for good_id in ["anne", "bronya", "hatsune-miku", "miki"]:
            persona = get_persona(good_id, "vi")
            assert persona["persona_id"] == good_id

    def test_cache_not_polluted_by_invalid(self):
        """A rejected id must not disturb or grow the cache.

        The direct `_persona_cache[...]` access is the point of this test, and it
        is why the loader keys the cache by plain slug and flattens the language
        on read instead of keying on (slug, lang). See `_load_persona`.
        """
        get_persona("anne", "vi")
        cached = _persona_cache.get("anne")
        assert cached is not None
        assert cached["persona_id"] == "anne"

        with pytest.raises(PersonaError):
            get_persona("../../etc/passwd", "vi")

        # The valid entry must NOT be corrupted by the invalid lookup
        assert _persona_cache["anne"] is cached

        # A bad id is never cached — a flood of distinct ones cannot grow it
        assert "../../etc/passwd" not in _persona_cache

        # And it stays refused on a second attempt
        with pytest.raises(PersonaError):
            get_persona("../../etc/passwd", "vi")

    def test_language_is_validated_like_the_slug(self):
        """`lang` also becomes part of a path, so it gets the same treatment."""
        for bad_lang in ["../vi", "vi/../en", "fr", ""]:
            with pytest.raises(PersonaError):
                get_persona("anne", bad_lang)
