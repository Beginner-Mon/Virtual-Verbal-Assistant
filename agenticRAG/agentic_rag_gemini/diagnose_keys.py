"""Diagnostic v2 — tests keys using exactly the fallback path in gemini_client.py."""
import sys, time
from pathlib import Path

# ── Load keys from .env ────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
keys, model = [], "gemini-2.5-flash"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line.startswith("GEMINI_API_KEYS="):
        keys = [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
    elif line.startswith("ORCHESTRATOR_MODEL="):
        model = line.split("=", 1)[1].strip()

print(f"Model : {model}")
print(f"Keys  : {len(keys)} found")
print("=" * 60)
sys.stdout.flush()

import google.generativeai as genai

for i, key in enumerate(keys, 1):
    label = f"Key {i}/{len(keys)}  ({key[:20]}...)"
    print(f"\n[{label}]")
    sys.stdout.flush()
    try:
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model_name=model)
        # Use the OLD API path (genai.types.GenerationConfig) — same as the fallback in gemini_client.py
        resp = m.generate_content(
            contents=[{"role": "user", "parts": [{"text": "Say OK in one word"}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                candidate_count=1,
                max_output_tokens=16,
            )
        )
        if not resp.candidates:
            print(f"  ❌  No candidates returned (prompt_feedback={getattr(resp, 'prompt_feedback', '?')})")
            sys.stdout.flush()
            continue

        candidate = resp.candidates[0]
        finish    = getattr(candidate, "finish_reason", "?")
        try:
            text = candidate.content.parts[0].text
        except Exception:
            text = "(no text)"

        if finish not in (1, "STOP"):
            print(f"  ⚠️  finish_reason={finish}  (non-STOP) — would trigger rotation")
        else:
            print(f"  ✅  finish_reason={finish}  text={text!r}")
        sys.stdout.flush()

    except Exception as e:
        print(f"  ❌  {type(e).__name__}: {e}")
        sys.stdout.flush()
    time.sleep(0.5)   # avoid hammering the API

print("\n" + "=" * 60)
print("Done.")
