-- Migration 001: Normalize messages out of conversations.messages JSONB
-- into a dedicated messages table (one row per message).
--
-- Run order: see PHASE-6.10-PREDEPLOY.md Task 1e.

-- Step 1: create messages table
--   intent / tokens / grader_result are fixed fields always present on
--   assistant rows → dedicated columns (not JSONB). NULL on user rows.
CREATE TABLE IF NOT EXISTS messages (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role           TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content        TEXT NOT NULL,
    intent         TEXT,          -- assistant only, NULL for user rows
    tokens         INT,           -- assistant only
    grader_result  TEXT,          -- assistant only: 'pass' | 'pass_with_warning' | 'retry'
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages (session_id, created_at);

-- Step 2: add migration flag so backfill script is idempotent
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS _migrated BOOLEAN DEFAULT false;

-- Step 3: note — DROP COLUMN messages chạy SAU khi test xanh (xem cuối Task 1)
