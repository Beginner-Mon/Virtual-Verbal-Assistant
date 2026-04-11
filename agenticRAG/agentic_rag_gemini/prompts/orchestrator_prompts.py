"""Orchestrator prompt templates for local Qwen2.5-3B model."""

# Compact prompt — ~160 tokens.
# Qwen2.5-3B processes input tokens serially; cutting 80% of the prompt
# reduces routing latency from ~15s to ~2-3s with no accuracy loss for
# the simple classification task.
ORCHESTRATOR_PROMPT = """\
Route the user query. Return ONLY JSON, no other text.

INTENT CLASSIFICATION:

greeting|followup_question|resume_conversation 
→ needs_retrieval=false, needs_memory=false, agents=["memory_agent"]

ask_exercise_info|general_fitness_question 
→ needs_retrieval=true, needs_memory=false, agents=["retrieval_agent"]

exercise_recommendation 
(User asks: "give me exercises for X" / "suggest exercises for Y")
→ needs_retrieval=true, needs_memory=true, needs_motion=false, agents=["retrieval_agent","memory_agent"]
→ Memory helps personalize recommendations based on user history/injuries

visualize_motion 
(User asks: "show me how to do X" / "visualize X" / "demonstrate X exercise")
→ needs_motion=true, needs_retrieval=true, needs_memory=false, agents=["retrieval_agent", "motion_agent"]

unknown → agents=["retrieval_agent"]

JSON schema:
{"intent":"...","exercise":null,"agents":[...],"needs_motion":false,"needs_retrieval":false,"needs_memory":false,"needs_web_search":false,"confidence":0.0}

exercise: name string if user mentions an exercise, else null.
needs_memory: true ONLY for personalized advice (exercise recommendations, injury-aware suggestions).
needs_web_search: always false (no web access).
confidence: 0.0-1.0.

KEY DISTINCTION:
- "exercises for neck pain" → exercise_recommendation (needs_motion=false, needs_memory=true)
- "how to do a chin tuck" → visualize_motion (needs_motion=true, needs_memory=false)
- "show me the squat" → visualize_motion (needs_motion=true, needs_memory=false)
- "show me how to stretch" → visualize_motion (needs_motion=true, needs_memory=false)
- "how to do a push up" → visualize_motion (needs_motion=true, needs_memory=false)
- "show me 5 exercises for fat loss" → exercise_recommendation (needs_motion=false, needs_memory=true)
- "show me some exercises for back pain" → exercise_recommendation (needs_motion=false, needs_memory=true)
- "what exercises help with stress" → exercise_recommendation (needs_motion=false, needs_memory=true)
- "what is a rotator cuff" → ask_exercise_info (needs_retrieval=true, needs_memory=false)

RULE: "show me how to [VERB/SINGLE EXERCISE]" → visualize_motion.
      "show me [NUMBER/PLURAL] exercises for [CONDITION]" → exercise_recommendation.
"""


