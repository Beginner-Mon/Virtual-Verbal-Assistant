# Final Implementation Plan: Session Management API

This plan outlines how we will add full chat session management natively into the backend, utilizing the existing `SessionStore` JSON-on-disk database system. 

By pushing this logic to the backend, we allow external apps (web/mobile) to easily construct UIs with complex sidebar histories simply by referencing `session_id`s, without needing to manually cache or push massive conversation arrays.

## User Review Required

> [!IMPORTANT]
> The endpoints listed below have been grouped into the base `api_server.py` codebase to give them direct access to `SessionStore` and the Vector database (`ChromaDB`). 
> Then, they will be seamlessly proxied directly through `main_api.py` (port 8080) so that your frontend only relies on one URL as normal.
> 
> Please review the exact API endpoints and the Query Integration logic. If everything looks good, approve the plan and I will begin the implementation!

## Proposed Changes

### 1. Pydantic Models & Typings 
Updates to `agenticRAG/agentic_rag_gemini/api_server.py` and `main_api.py`.

*   **Update `QueryRequest` / `AnswerRequest`**: Add the `session_id: Optional[str] = None` field.
*   **Create `SessionCreateRequest`**: Accepts `user_id: str` and `first_message: Optional[str]`.
*   **Create `SessionMetaResponse`**: Returns lightweight session info (title, timestamp) used for listing history sidebars.

---

### 2. Core API Endpoints (in `api_server.py`)
These are the new session management endpoints. They directly interface with `SessionStore`.

#### `POST /sessions`
*   **Action**: Initializes a `SessionStore(user_id)`. Calls `.create_session(first_message)`.
*   **Returns**: `{"session_id": "...", "user_id": "...", "title": "..."}`

#### `GET /sessions/{user_id}`
*   **Action**: Initializes a `SessionStore(user_id)`. Calls `.list_sessions()`.
*   **Returns**: An array of `SessionMetaResponse` showing all past chats ordered by newest first.

#### `GET /sessions/{user_id}/{session_id}`
*   **Action**: Initializes a `SessionStore(user_id)`. Calls `.load_session(session_id)`.
*   **Returns**: The full JSON object including the `"messages": []` array with all the past conversation turns.

#### `DELETE /sessions/{user_id}/{session_id}`
*   **Action**: Calls `.delete_session(session_id)`.
*   **Returns**: Status 200 confirming the chat was deleted visually.

#### `POST /sessions/{user_id}/{session_id}/summarize`
*   **Action**: A utility endpoint for the frontend. When a user closes or changes a session, the frontend calls this. The backend triggers the `SummarizeAgent` to safely bundle the session data and insert the summary into the `ChromaDB` vector search memory.

---

### 3. Query Flow Integration (in `api_server.py`)
Modifications within `_run_query_task()` / `process_query()`:

1.  **Intercept**: Before calling AgenticRAG, check if `session_id` exists in the payload.
2.  **Load History**: If `session_id` provided, invoke `SessionStore(user_id).load_session(session_id)`. Extract the `"messages"` array and map them seamlessly to the `history` variable.
3.  **Execute**: Run LLM and RAG components normally using this populated history.
4.  **Save State**: Before issuing the final HTTP response to the client, call `SessionStore.save_turn()` to save the `user`'s prompt, and call it again to save the `assistant`'s generated text answer.

---

### 4. API Proxying (in `main_api.py`)
Since your `main_api.py` operates as an intelligent gateway (port 8080) for `DART`, `SpeechLLM`, and `AgenticRAG`, we need to route the session queries through it.

*   Update `AnswerRequest` to accept `session_id`.
*   Create `@app.post("/sessions")`, `@app.get("/sessions/{user_id}")`, `@app.get("/sessions/{user_id}/{session_id}")`, `@app.delete("/sessions/{user_id}/{session_id}")` endpoints inside `main_api.py`.
*   Use `await _proxy_main_api_request("GET", f"/sessions/{user_id}")` to silently bounce the API traffic straight to `api_server.py`.

## Verification Plan

### Automated / Manual Fast Testing
1. Use `curl` or Postman to hit `POST http://localhost:8080/sessions` and verify a `session_id` is returned.
2. POST a query to `http://localhost:8080/answer` including the new `session_id`. Ensure the RAG process completes correctly.
3. Fetch `GET http://localhost:8080/sessions/{user_id}/{session_id}` and confirm that the dialogue generated in step 2 was written automatically into the array.
4. Ensure no existing `main_api.py` functionality is broken for clients *not* sending `session_id`.
