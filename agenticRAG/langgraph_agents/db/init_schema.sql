-- Phase 1 PostgreSQL schema for LangGraph agent system
-- Run: python -m langgraph_agents.db.init_schema

CREATE EXTENSION IF NOT EXISTS vector;

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Zero-cost internal access. Stripe sandbox state is informational and never
-- changes access_plan to a paid entitlement.
CREATE TABLE IF NOT EXISTS billing_accounts (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    access_plan TEXT NOT NULL DEFAULT 'DEMO' CHECK (access_plan IN ('FREE', 'DEMO')),
    stripe_customer_id TEXT UNIQUE,
    stripe_subscription_id TEXT UNIQUE,
    stripe_subscription_status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS billing_webhook_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    livemode BOOLEAN NOT NULL DEFAULT false CHECK (livemode = false),
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Conversations (session store)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_id UUID NOT NULL,
    messages JSONB DEFAULT '[]',
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT conversations_session_id_unique UNIQUE (session_id)
);

-- Documents (uploaded files metadata)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    filename TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,
    source_id UUID,
    content TEXT,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations (session_id);
