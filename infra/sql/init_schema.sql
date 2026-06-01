CREATE EXTENSION IF NOT EXISTS vector;

-- Users
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile     JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Conversations (session header — không còn cột messages JSONB)
CREATE TABLE IF NOT EXISTS conversations (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id),
    summary     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Messages (one row per message — thay cho conversations.messages JSONB)
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

-- Documents (uploaded files metadata)
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id),
    filename    TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type  TEXT NOT NULL,
    source_id    UUID,
    content      TEXT,
    embedding    vector(384),
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embeddings_source       ON embeddings (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user      ON conversations (user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session   ON conversations (session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_pagination ON messages (session_id, created_at DESC);