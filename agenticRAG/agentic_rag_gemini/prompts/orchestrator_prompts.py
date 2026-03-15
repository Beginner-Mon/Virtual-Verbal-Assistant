"""Orchestrator prompt templates for local Qwen2.5-3B model."""

# Compact prompt — ~160 tokens.
# Qwen2.5-3B processes input tokens serially; cutting 80% of the prompt
# reduces routing latency from ~15s to ~2-3s with no accuracy loss for
# the simple classification task.
ORCHESTRATOR_PROMPT = """\
Route the user query. Return ONLY JSON, no other text.

INTENTS (pick one):
greeting|followup_question|resume_conversation → needs_retrieval=false, agents=["memory_agent"]
ask_exercise_info|general_fitness_question → needs_retrieval=true, agents=["retrieval_agent"]
visualize_motion → needs_motion=true, needs_retrieval=true, agents=["retrieval_agent","motion_agent"]
unknown → agents=["retrieval_agent"]

JSON schema:
{"intent":"...","exercise":null,"agents":[...],"needs_motion":false,"needs_retrieval":false,"needs_web_search":false,"confidence":0.0}

exercise: name string if user mentions an exercise, else null.
needs_web_search: always false (no web access).
confidence: 0.0-1.0."""

