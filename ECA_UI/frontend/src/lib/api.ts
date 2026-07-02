/**
 * VVA API client — PART B auth integration spec.
 *
 * - API_BASE from VITE_API_BASE_URL env (default: http://localhost:8080)
 * - authHeader(): attach Cognito idToken when Amplify is configured + user signed in
 * - currentUserId(): Cognito sub when authed, else stable demo UUID from localStorage
 * - streamChat(): real SSE via ReadableStream.getReader() with fallback to text()
 *   (SSE parsing logic ported from ECA_UI/test-ui/sse-test/api.js ~line 506)
 * - Sessions + user_memory CRUD helpers
 */

import { fetchAuthSession } from 'aws-amplify/auth'

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8080'

// ── Auth helpers ──────────────────────────────────────────────────────────────

/** Returns { Authorization: 'Bearer <idToken>' } when signed in, else {}. */
async function authHeader(): Promise<Record<string, string>> {
  try {
    const session = await fetchAuthSession()
    const token = session.tokens?.idToken?.toString()
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

/** Stable user id: Cognito sub when authed, else a persistent demo UUID. */
async function currentUserId(): Promise<string> {
  try {
    const session = await fetchAuthSession()
    const sub = session.tokens?.idToken?.payload?.sub
    if (sub && typeof sub === 'string') return sub
  } catch {
    // not signed in — fall through to demo id
  }

  const DEMO_KEY = 'vva_demo_user'
  let demoId = localStorage.getItem(DEMO_KEY)
  if (!demoId) {
    demoId = crypto.randomUUID()
    localStorage.setItem(DEMO_KEY, demoId)
  }
  return demoId
}

// ── SSE parser (ported from test-ui/sse-test/api.js _parseSSEBlocks) ─────────

type SSEEventCallback = (eventType: string, data: unknown) => void

function _parseSSEBlocks(text: string, emit: SSEEventCallback): void {
  for (const block of text.split(/\r?\n\r?\n/)) {
    if (!block.trim()) continue
    let eventType: string | null = null
    let dataStr = ''
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith(':')) continue // comment / heartbeat
      if (line.startsWith('event:')) eventType = line.slice(6).trim()
      else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
    }
    if (!eventType) continue
    try {
      emit(eventType, dataStr ? JSON.parse(dataStr) : {})
    } catch (e) {
      console.warn('[SSE] Failed to parse data:', dataStr, e)
    }
  }
}

// ── streamChat ────────────────────────────────────────────────────────────────

export interface StreamChatOptions {
  query: string
  sessionId: string
  personaId?: string
  outputMode?: 'text' | 'speech' | 'both'
  webSearch?: boolean
}

/**
 * POST /chat and stream SSE events back via onEvent callback.
 *
 * Uses ReadableStream.getReader() for true streaming; falls back to text()
 * parsing (with per-token delay) when a proxy/AV has buffered the response.
 *
 * @param options   Chat request parameters
 * @param onEvent   Callback invoked for each SSE event (eventType, data)
 * @param signal    Optional AbortController signal to cancel mid-stream
 */
export async function streamChat(
  options: StreamChatOptions,
  onEvent: SSEEventCallback,
  signal?: AbortSignal,
): Promise<void> {
  const { query, sessionId, personaId = 'eca_default', outputMode = 'text', webSearch = false } =
    options

  const userId = await currentUserId()
  const extraHeaders = await authHeader()

  const resp = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...extraHeaders,
    },
    body: JSON.stringify({
      query,
      user_id: userId,
      session_id: sessionId,
      persona_id: personaId,
      output_mode: outputMode,
      web_search: webSearch,
    }),
    signal,
  })

  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await resp.text()}`)
  }

  // Clone so fallback text() path can still read the body
  const respClone = resp.clone()

  const reader = resp.body?.getReader?.() ?? null
  if (reader) {
    const decoder = new TextDecoder()
    let buffer = ''
    let emittedAny = false
    const wrappedEmit: SSEEventCallback = (t, d) => {
      emittedAny = true
      onEvent(t, d)
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lastBoundary = buffer.lastIndexOf('\n\n')
        if (lastBoundary === -1) continue
        const ready = buffer.slice(0, lastBoundary)
        buffer = buffer.slice(lastBoundary + 2)
        _parseSSEBlocks(ready, wrappedEmit)
      }
      if (buffer.trim()) _parseSSEBlocks(buffer, wrappedEmit)
      if (emittedAny) return // streaming worked
      console.warn('[SSE] reader returned 0 events — falling back to text()')
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e
      console.warn('[SSE] reader failed, falling back to text():', e)
    }
  }

  // Fallback: body was already buffered server/proxy-side
  const fullBody = await respClone.text()
  const events: Array<{ type: string; data: unknown }> = []
  _parseSSEBlocks(fullBody, (t, d) => events.push({ type: t, data: d }))

  const TYPING_DELAY_MS = 15
  for (const e of events) {
    onEvent(e.type, e.data)
    if (e.type === 'token') {
      await new Promise<void>((r) => setTimeout(r, TYPING_DELAY_MS))
    }
  }
}

// ── Session CRUD ───────────────────────────────────────────────────────────────

export async function listSessions() {
  const userId = await currentUserId()
  const headers = await authHeader()
  const resp = await fetch(
    `${API_BASE}/sessions?user_id=${encodeURIComponent(userId)}`,
    { headers },
  )
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function getSession(sessionId: string) {
  const userId = await currentUserId()
  const headers = await authHeader()
  const resp = await fetch(
    `${API_BASE}/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { headers },
  )
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function deleteSession(sessionId: string) {
  const userId = await currentUserId()
  const headers = await authHeader()
  const resp = await fetch(
    `${API_BASE}/sessions/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE', headers },
  )
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

// ── User memory CRUD ───────────────────────────────────────────────────────────

export async function listUserMemory() {
  const userId = await currentUserId()
  const headers = await authHeader()
  const resp = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}/memory`, { headers })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function createUserMemory(factText: string, category?: string) {
  const userId = await currentUserId()
  const extraHeaders = await authHeader()
  const resp = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    body: JSON.stringify({ fact_text: factText, category }),
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function deleteUserMemory(factId: string) {
  const userId = await currentUserId()
  const extraHeaders = await authHeader()
  const resp = await fetch(
    `${API_BASE}/users/${encodeURIComponent(userId)}/memory/${encodeURIComponent(factId)}`,
    { method: 'DELETE', headers: { ...extraHeaders } },
  )
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}
