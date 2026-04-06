"""Minimal unit test — no logging, no imports that trigger logger."""
import re
import sys

# Inline the functions we need to test (avoid importing the full module
# which triggers SentenceTransformer / ChromaDB initialization).

_FALLBACK_DESCRIPTION = "a person performs a controlled full body exercise movement"

FORCE_EXPANSION_LIST = [
    "burpee", "turkish get up", "turkish getup",
    "clean and jerk", "clean and press",
    "snatch", "thruster", "man maker", "man-maker",
    "devil press", "muscle up", "muscle-up",
]

def _needs_force_expansion(query):
    query_lower = query.lower()
    return any(term in query_lower for term in FORCE_EXPANSION_LIST)

def _fallback_parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    for prefix in (
        "kinematic description:", "output:", "rewritten caption:",
        "description:", "motion:", "caption:", "refined prompt:",
        "sure!", "sure,", "here is", "here's",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    lines = text.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("chosen path") or lower.startswith("optimized prompt"):
            continue
        if lower.startswith("path:") or lower.startswith("strategy:"):
            continue
        if lower.startswith("step ") and ":" in lower:
            continue
        cleaned_lines.append(line.strip())
    text = " ".join(cleaned_lines).strip()
    person_match = re.search(r"(a person\b[^.!?\n]*[.!?]?)", text, re.IGNORECASE)
    if person_match:
        found = person_match.group(1).strip().rstrip(".")
        if len(found.split()) >= 5:
            return found
    if len(text.split()) >= 5:
        return text
    return _FALLBACK_DESCRIPTION

def _ensure_quality(text):
    text = text.strip()
    if not text:
        return _FALLBACK_DESCRIPTION
    text = re.sub(r"\[\[|\]\]", "", text).strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    lower = text.lower()
    if not lower.startswith("a person"):
        text = f"a person {text}"
    words = text.split()
    if len(words) < 5:
        return _FALLBACK_DESCRIPTION
    if len(words) > 30:
        text = " ".join(words[:30])
    return text

def _extract_and_validate(raw_text):
    raw_text = raw_text.strip()
    match = re.search(r"\[\[\s*(.*?)\s*\]\]", raw_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted.split()) >= 5:
            return _ensure_quality(extracted)
    cleaned = _fallback_parse(raw_text)
    return _ensure_quality(cleaned)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST SUITE: _extract_and_validate")
print("=" * 70)

cases = [
    ('[[ a person jumps up and raises both arms overhead ]]',
     "a person jumps up"),
    ('Sure! Here is your result:\n[[ a person bends their knees into a squat ]]',
     "a person bends their knees"),
    ('Step 1: Extract.\nStep 2: Check.\n[[ a person walks forward at a steady pace ]]',
     "a person walks forward"),
    # Unclosed bracket — fallback parse finds "a person raises arms..."
    ('[[ a person raises arms overhead and lowers them',
     "a person raises arms"),
    # No brackets — fallback parse finds the sentence directly
    ('a person performs a controlled lunge stepping forward with right foot',
     "a person performs a controlled lunge"),
    # Empty brackets — fallback returns default
    ('[[ ]]', _FALLBACK_DESCRIPTION[:15]),
    # Too short — fallback returns default
    ('[[ a person ]]', _FALLBACK_DESCRIPTION[:15]),
]

passed = 0
for i, (raw, expected_snippet) in enumerate(cases, 1):
    result = _extract_and_validate(raw)
    ok = expected_snippet.lower() in result.lower()
    status = "PASS" if ok else "FAIL"
    print(f"\n  Test {i}: {status}")
    print(f"    Input:    {raw[:70]}")
    print(f"    Output:   {result}")
    print(f"    Expected: contains '{expected_snippet}'")
    if ok:
        passed += 1

print(f"\n  Extraction Results: {passed}/{len(cases)} passed")

print("\n" + "=" * 70)
print("TEST SUITE: Force-Expansion Detection")
print("=" * 70)

fe_cases = [
    ("Show me a burpee", True),
    ("How to do a turkish get up", True),
    ("clean and jerk technique", True),
    ("Show me a squat", False),
    ("How to do jumping jacks", False),
    ("muscle up tutorial", True),
]

fe_passed = 0
for query, expected in fe_cases:
    result = _needs_force_expansion(query)
    ok = result == expected
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  '{query}' -> expand={result} (expected={expected})")
    if ok:
        fe_passed += 1

print(f"\n  Force-Expansion Results: {fe_passed}/{len(fe_cases)} passed")

print("\n" + "=" * 70)
all_ok = (passed == len(cases)) and (fe_passed == len(fe_cases))
if all_ok:
    print("ALL UNIT TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
