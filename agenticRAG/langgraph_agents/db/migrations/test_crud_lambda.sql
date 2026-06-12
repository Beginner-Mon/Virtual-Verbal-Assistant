-- ============================================================================
-- Manual test harness for the CRUD functions that become Lambda in Phase 7.
--
-- Covers the exact queries run by (db/session_store.py + api/main.py):
--   list_user_sessions      → GET  /sessions?user_id=...
--   load_session_messages   → GET  /sessions/{id}  (+ cursor pagination)
--   delete_session          → DELETE /sessions/{user_id}/{session_id}
--   write_session_turn      → (inserts seeded below so the reads have data)
--
-- Prereq: init_schema.sql + 001_normalize_messages.sql already applied.
--
-- Usage:
--   psql $DATABASE_URL -f .../test_crud_lambda.sql
--   (or paste section-by-section into a psql shell)
--
-- NOTE on user_id: the app sends plain strings ("test-user"); session_store
-- coerces them via uuid5(NAMESPACE_DNS, value) before hitting the DB. The UUIDs
-- below are the PRE-COMPUTED uuid5 values so seed + query match exactly:
--     'test-user' -> a5f4fe64-dd02-5776-b5f9-59fccda849ee
--     'anonymous' -> 204ae2e2-c270-529f-bf19-28e895e20a16
-- If you test a different user string, recompute with:
--     python -c "import uuid; print(uuid.uuid5(uuid.NAMESPACE_DNS,'YOUR-STRING'))"
-- ============================================================================

\set ON_ERROR_STOP on

BEGIN;

-- ── Pinned IDs so every statement below is reproducible ──────────────────────
\set test_user      '''a5f4fe64-dd02-5776-b5f9-59fccda849ee'''
\set session_a      '''11111111-1111-1111-1111-111111111111'''
\set session_b      '''22222222-2222-2222-2222-222222222222'''

-- ============================================================================
-- 0. SEED  (mirrors write_session_turn: upsert user, upsert conversation, 2 msgs)
-- ============================================================================

INSERT INTO users (id) VALUES (:test_user::uuid)
ON CONFLICT (id) DO NOTHING;

-- Session A: 2 turns (4 messages), updated more recently
INSERT INTO conversations (id, user_id, session_id, created_at, updated_at)
VALUES (gen_random_uuid(), :test_user::uuid, :session_a::uuid,
        now() - interval '2 hours', now() - interval '5 minutes')
ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at;

INSERT INTO messages (session_id, role, content, intent, tokens, grader_result, created_at) VALUES
  (:session_a::uuid, 'user',      'bài tập cho đau lưng',          NULL,                      NULL, NULL,   now() - interval '120 min'),
  (:session_a::uuid, 'assistant', 'Có 3 bài tập tốt cho lưng...',  'exercise_recommendation', 842,  'pass', now() - interval '119 min'),
  (:session_a::uuid, 'user',      'bài nào dễ nhất?',              NULL,                      NULL, NULL,   now() - interval '118 min'),
  (:session_a::uuid, 'assistant', 'Bird-dog là bài nhẹ nhất...',   'exercise_recommendation', 510,  'pass', now() - interval '117 min');

-- Session B: 1 turn (2 messages), older — checks ORDER BY updated_at DESC
INSERT INTO conversations (id, user_id, session_id, created_at, updated_at)
VALUES (gen_random_uuid(), :test_user::uuid, :session_b::uuid,
        now() - interval '1 day', now() - interval '1 day')
ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at;

INSERT INTO messages (session_id, role, content, intent, tokens, grader_result, created_at) VALUES
  (:session_b::uuid, 'user',      'xin chào',          NULL,           NULL, NULL,   now() - interval '1 day'),
  (:session_b::uuid, 'assistant', 'Chào bạn! ...',     'conversation', 60,   'pass', now() - interval '1 day' + interval '1 sec');

-- ============================================================================
-- 1. list_user_sessions(user_id, limit=50)
--    Expect: 2 rows, session_a first (updated_at DESC).
--    first_user_message_preview = first 'user' message per session.
--    message_count = total rows per session (A=4, B=2).
-- ============================================================================
\echo '--- TEST 1: list_user_sessions ---'

SELECT c.session_id::text,
       c.created_at,
       c.updated_at,
       COALESCE(first_msg.content, '(empty)')  AS first_user_message_preview,
       COALESCE(msg_count.cnt, 0)::int          AS message_count
FROM conversations c
LEFT JOIN LATERAL (
    SELECT content FROM messages
    WHERE session_id = c.session_id AND role = 'user'
    ORDER BY created_at LIMIT 1
) first_msg ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::int AS cnt FROM messages
    WHERE session_id = c.session_id
) msg_count ON true
WHERE c.user_id = :test_user::uuid
ORDER BY c.updated_at DESC
LIMIT 50;

-- ============================================================================
-- 2. load_session_messages — header check (returns NULL/404 path if missing)
--    Expect: 1 row for session_a.
-- ============================================================================
\echo '--- TEST 2a: load_session_messages header (exists) ---'
SELECT updated_at FROM conversations
WHERE user_id = :test_user::uuid AND session_id = :session_a::uuid;

\echo '--- TEST 2b: header for a NON-existent session (expect 0 rows = 404) ---'
SELECT updated_at FROM conversations
WHERE user_id = :test_user::uuid
  AND session_id = '99999999-9999-9999-9999-999999999999'::uuid;

-- ============================================================================
-- 3. load_session_messages — message page (no cursor, limit=50)
--    App fetches DESC then reverses in Python → here ORDER BY DESC to mirror
--    the raw query. Expect 4 rows for session_a.
-- ============================================================================
\echo '--- TEST 3: load_session_messages page (no cursor) ---'
SELECT role, content, intent, tokens, grader_result, created_at
FROM messages
WHERE session_id = :session_a::uuid
ORDER BY created_at DESC
LIMIT 50;

-- ============================================================================
-- 4. Cursor pagination (before = created_at of oldest currently shown).
--    Simulate "scroll up": take cursor = the 3rd message's timestamp, expect
--    only messages strictly older than it. Uses the covering index
--    idx_messages_session_created (session_id, created_at).
-- ============================================================================
\echo '--- TEST 4: cursor pagination (before cursor) ---'
WITH cursor AS (
    SELECT created_at AS ts FROM messages
    WHERE session_id = :session_a::uuid
    ORDER BY created_at DESC
    OFFSET 2 LIMIT 1            -- pretend we already showed the 2 newest
)
SELECT m.role, m.content, m.created_at
FROM messages m, cursor
WHERE m.session_id = :session_a::uuid
  AND m.created_at < cursor.ts
ORDER BY m.created_at DESC
LIMIT 50;

-- ============================================================================
-- 5. EXPLAIN — confirm the page query uses the index, not a seq scan.
--    Look for "Index Scan using idx_messages_session_created" in output.
-- ============================================================================
\echo '--- TEST 5: EXPLAIN page query (want Index Scan) ---'
EXPLAIN (ANALYZE, BUFFERS)
SELECT role, content, intent, tokens, grader_result, created_at
FROM messages
WHERE session_id = :session_a::uuid
ORDER BY created_at DESC
LIMIT 50;

-- ============================================================================
-- 6. delete_session(user_id, session_id)
--    App runs DELETE on conversations; messages cascade via FK
--    (ON DELETE CASCADE). Verify both gone.
-- ============================================================================
\echo '--- TEST 6a: row counts BEFORE delete (conv should be 2, msgs 6) ---'
SELECT
  (SELECT COUNT(*) FROM conversations WHERE user_id = :test_user::uuid) AS conversations,
  (SELECT COUNT(*) FROM messages
     WHERE session_id IN (:session_a::uuid, :session_b::uuid))          AS messages;

\echo '--- TEST 6b: delete session_b ---'
DELETE FROM conversations
WHERE user_id = :test_user::uuid AND session_id = :session_b::uuid;

\echo '--- TEST 6c: verify cascade — session_b messages gone (expect 0) ---'
SELECT COUNT(*) AS session_b_messages_remaining
FROM messages WHERE session_id = :session_b::uuid;

\echo '--- TEST 6d: session_a untouched (expect 4) ---'
SELECT COUNT(*) AS session_a_messages
FROM messages WHERE session_id = :session_a::uuid;

-- ============================================================================
-- Leave everything in a transaction. ROLLBACK = throwaway test (default below).
-- Change to COMMIT if you want the seed data to persist for further poking.
-- ============================================================================
ROLLBACK;
-- COMMIT;
