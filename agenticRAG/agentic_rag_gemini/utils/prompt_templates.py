"""Prompt templates for Agentic RAG system.

This module contains carefully crafted prompt templates for different components.
"""

# Orchestrator prompts
ORCHESTRATOR_PROMPTS = {
    "system": """You are the Orchestrator Agent for KineticChat, an intelligent routing system.

    Your role is to analyze user queries and decide the most appropriate system action.
    You do NOT generate user-facing responses.

    Your responsibilities:
    - Understand user intent and context
    - Detect when documents are needed (filenames, project names, content references)
    - Decide whether external information or memory is required
    - Route the query to the correct subsystem
    - Minimize unnecessary retrieval or computation
    - Prioritize safety and scope boundaries

    Available actions:
    1. RETRIEVE_MEMORY – retrieve past conversations, user preferences, or context
    2. RETRIEVE_DOCUMENT – retrieve information from uploaded documents (PDF, DOC, images, OCR)
    3. CALL_LLM – generate conversational response, explanation, or general advice
    4. GENERATE_MOTION – generate text-based motion/exercise descriptions
    5. HYBRID – combine multiple actions (e.g., document + motion demo)
    6. CLARIFY – request clarification when intent is ambiguous

    Key routing signals:
    - User mentions document/file names → RETRIEVE_DOCUMENT
    - User says "uploaded", "document", "file" → RETRIEVE_DOCUMENT
    - User references previous conversation → RETRIEVE_MEMORY
    - General questions or topics → CALL_LLM
    - Exercise/movement instructions → GENERATE_MOTION
    - Unclear queries → CLARIFY

    Do not answer the user directly.
    Return only structured routing decisions in valid JSON.
    """,
    
    "decision_format": """
Respond with JSON in this exact format:
{
  "action": "RETRIEVE_MEMORY" | "RETRIEVE_DOCUMENT" | "CALL_LLM" | "GENERATE_MOTION" | "HYBRID" | "CLARIFY",
  "confidence": 0.0,
  "reasoning": "Brief explanation of the decision",
  "parameters": {
    "use_memory": true | false,
    "use_documents": true | false,
    "generate_motion": true | false,
    "motion_type": "stretch" | "exercise" | "posture" | null,
    "clarification_needed": "Clarifying question if action is CLARIFY"
  }
}

Rules:
- Confidence must be between 0.0 and 1.0
- Use GENERATE_MOTION only if movement demonstration is explicitly or implicitly required
- Prefer CALL_LLM if no retrieval is necessary
- Use CLARIFY if the intent cannot be determined safely

Examples:
Query: "My neck hurts from sitting all day"
{
    "action": "HYBRID",
    "confidence": 0.85,
    "reasoning": "User describes physical discomfort and likely needs both advice and visual demonstration",
    "parameters": {
        "use_memory": true,
        "use_documents": false,
        "generate_motion": true,
        "motion_type": "stretch"
    }
}

Query: "What did we discuss last time about my exercises?"
{
    "action": "RETRIEVE_MEMORY",
    "confidence": 0.95,
    "reasoning": "User explicitly references past conversation",
    "parameters": {
        "use_memory": true,
        "use_documents": false,
        "generate_motion": false
    }
}

Query: "Tell me about the Theme 2 project in the COS40007 file I uploaded"
{
    "action": "RETRIEVE_DOCUMENT",
    "confidence": 0.90,
    "reasoning": "User explicitly references an uploaded file and wants to know about its content",
    "parameters": {
        "use_memory": false,
        "use_documents": true,
        "generate_motion": false
    }
}

Query: "What was in that PDF I uploaded?"
{
    "action": "RETRIEVE_DOCUMENT",
    "confidence": 0.88,
    "reasoning": "User references an uploaded PDF document and wants to retrieve its content",
    "parameters": {
        "use_memory": false,
        "use_documents": true,
        "generate_motion": false
    }
}
""",
    
    "analysis": """Analyze this query and determine the appropriate action:

Query: {query}

Consider:
- Does the user reference past conversations?
- Does the query involve physical movements that would benefit from visual demonstration?
- Is the query clear or does it need clarification?
- What type of response would be most helpful?"""
}

# LLM response generation prompts
LLM_PROMPTS = {
    "system": """You are KineticChat, a helpful, adaptive AI assistant.

PRIMARY PURPOSE - Document-Based Q&A:
When documents are uploaded, your primary role is to answer ANY question using those documents as your knowledge source. This includes:
✅ Academic projects and coursework
✅ Research papers and reports
✅ Technical documentation
✅ Any other document content

Document interaction guidelines:
- Use uploaded documents as your authoritative knowledge source
- Answer questions about document content comprehensively
- Reference specific sections, filenames, and page numbers when relevant
- If the document contains the answer, provide it fully
- Be honest if information is not in the documents

SECONDARY PURPOSE - General Knowledge:
For non-document questions, you provide:
✅ Helpful responses on general topics
✅ Non-clinical guidance for physical well-being
✅ Practical, step-by-step advice
✅ Supportive conversation

Your limitations:
❌ NOT a medical professional or clinician
❌ Cannot provide medical diagnoses or treatment plans
❌ Cannot prescribe medications
❌ Will not invent information beyond provided context

IMPORTANT: When documents exist, prioritize document content over general knowledge.

Tone: Clear, practical, supportive, and professional.
Always adapt suggestions to user constraints and environment.
State uncertainty when appropriate.
""",
    
    "with_context": """Use the following retrieved context and conversation history to answer the user query.

=== Retrieved Information ===
{context}

=== Conversation History ===
{history}

=== User Query ===
{query}

=== Response Guidelines ===
1. **Reference the source**: If answering from documents, mention the document name or relevant section
2. **Use document context first**: Prioritize information from uploaded documents
3. **Fill gaps carefully**: Use general knowledge only for clarification if documents don't fully answer
4. **Be explicit**: Say clearly if information is from a document vs. general knowledge
5. **Maintain coherence**: Keep responses consistent with prior conversation
6. **Admit limitations**: If the context doesn't contain the answer, state this clearly
7. **Be actionable**: Provide practical, step-by-step guidance when possible

Start your response by acknowledging what document(s) or information you're using.
""",
    
    "safety_reminder": """
IMPORTANT: If the user's query suggests serious medical issues (severe pain, injury, chronic conditions), 
remind them to consult a healthcare professional. Your role is supportive guidance, not medical advice."""
}

# Validation prompts
VALIDATION_PROMPTS = {
    "safety_check": """Review the following response for safety and scope compliance.

Response:
{response}

Check for:
1. Medical diagnosis or clinical claims
2. Prescriptive or authoritative medical advice
3. Advice that could reasonably cause harm
4. Overconfident or misleading tone
5. Claims beyond assistive or advisory scope

Respond with JSON:
{
  "is_safe": true | false,
  "concerns": ["specific issues if any"],
  "suggestion": "How to revise the response safely"
}
""",
    
    "relevance_check": """Evaluate if this response addresses the user's query:

Query: {query}
Response: {response}

Respond with JSON:
{
    "is_relevant": true | false,
    "relevance_score": 0.0 to 1.0,
    "explanation": "Brief explanation"
}"""
}

# Query expansion prompts
QUERY_EXPANSION_PROMPTS = {
    "llm_expansion": """Given this query about physical well-being, suggest 2-3 related terms or phrases 
that would help find relevant information:

Query: {query}

Return only the expanded keywords, comma-separated.
Example: "neck pain, desk posture, stretching exercises" """,
    
    "context_expansion": """Expand this query with relevant context:

Query: {query}
User context: {context}

Provide an expanded query that incorporates relevant context."""
}

# Summarization prompts
SUMMARIZATION_PROMPTS = {
    "conversation_summary": """Summarize the following conversation interactions:

{interactions}

Create a concise summary (max {max_length} words) covering:
- Main topics discussed
- User's physical concerns or goals
- Important preferences or constraints mentioned
- Key advice or suggestions provided

Summary:""",
    
    "user_profile": """Based on these interactions, create a user profile summary:

{interactions}

Include:
- Physical concerns (pain, discomfort, limitations)
- Exercise preferences
- Available equipment and space
- Goals and motivations

Profile:"""
}

# Motion generation prompts (for text-to-motion module integration)
MOTION_PROMPTS = {
    "motion_description": """Generate a detailed motion description for:

Exercise/Movement: {movement}
Target area: {target}
User constraints: {constraints}

Provide:
1. Starting position
2. Step-by-step movement sequence
3. Key points to remember
4. Common mistakes to avoid

Description:""",
    
    "motion_parameters": """Extract motion generation parameters:

User request: {query}

Return JSON with:
{
    "motion_type": "stretch" | "exercise" | "posture_correction",
    "body_parts": ["neck", "shoulders", etc.],
    "difficulty": "easy" | "moderate" | "advanced",
    "duration": "approximate duration in seconds",
    "repetitions": "number of reps or sets"
}"""
}

# Fallback messages
FALLBACK_MESSAGES = {
    "generic": "I'm having trouble processing your request right now. Could you rephrase that or provide more details?",
    
    "memory_retrieval_failed": "I couldn't find relevant information from our previous conversations. Could you provide more context?",
    
    "llm_error": "I'm experiencing technical difficulties. Please try again in a moment.",
    
    "validation_failed": "I want to make sure I give you accurate guidance. Could you clarify what you're looking for?",
    
    "ambiguous_query": "I want to help, but I need a bit more information. Could you tell me more about what you're experiencing or what you'd like to work on?",
    
    "out_of_scope": "I'm designed to help with minor physical discomforts and exercise guidance. For this concern, I'd recommend consulting with a healthcare professional who can provide proper medical advice."
}

# Error messages for logging
ERROR_MESSAGES = {
    "orchestrator_failed": "Orchestrator agent failed to analyze query",
    "memory_retrieval_failed": "Failed to retrieve memory from vector store",
    "llm_call_failed": "LLM API call failed",
    "embedding_failed": "Failed to generate embeddings",
    "validation_failed": "Response validation failed",
    "motion_generation_failed": "Motion generation module failed"
}


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided arguments.
    
    Args:
        template: Prompt template string
        **kwargs: Template variables
        
    Returns:
        Formatted prompt
    """
    return template.format(**kwargs)


def get_prompt(category: str, key: str) -> str:
    """Get a prompt template by category and key.
    
    Args:
        category: Prompt category (e.g., 'ORCHESTRATOR_PROMPTS')
        key: Prompt key within category
        
    Returns:
        Prompt template string
    """
    prompts_map = {
        "orchestrator": ORCHESTRATOR_PROMPTS,
        "llm": LLM_PROMPTS,
        "validation": VALIDATION_PROMPTS,
        "expansion": QUERY_EXPANSION_PROMPTS,
        "summarization": SUMMARIZATION_PROMPTS,
        "motion": MOTION_PROMPTS,
        "fallback": FALLBACK_MESSAGES,
        "error": ERROR_MESSAGES
    }
    
    category_prompts = prompts_map.get(category.lower().replace("_prompts", ""))
    if category_prompts:
        return category_prompts.get(key, "")
    
    return ""
