/**
 * VVA API client — PART B auth integration spec.
 *
 * - API_BASE from VITE_API_BASE_URL env (default: http://localhost:8080)
 * - authHeader(): attach Clerk session token when Clerk Development is configured
 * - currentUserId(): Clerk user id when authed, else stable demo UUID
 * - streamChat(): real SSE via ReadableStream.getReader() with fallback to text()
 *   (SSE parsing logic ported from ECA_UI/test-ui/sse-test/api.js ~line 506)
 * - Sessions + user_memory CRUD helpers
 */

import axios from 'axios'
import { clerkAuthHeader, clerkUserId } from './clerkAuth'

// ── Config ────────────────────────────────────────────────────────────────────

const _raw = import.meta.env.VITE_API_BASE_URL as string | undefined
if (!_raw) {
  throw new Error(
    'VITE_API_BASE_URL is not set. Create a .env.local file with:\n' +
    '  VITE_API_BASE_URL=http://localhost:8080'
  )
}
const API_BASE: string = _raw

// ── Auth helpers ──────────────────────────────────────────────────────────────

/** Returns { Authorization: 'Bearer <idToken>' } when signed in, else {}. */
async function authHeader(): Promise<Record<string, string>> {
  return clerkAuthHeader()
}

// ── Axios instance (REST only) ──────────────────────────────────────────────────
// Used for non-streaming CRUD (sessions, user_memory). The request interceptor
// attaches the Cognito idToken once, so callers don't repeat auth wiring; axios
// also throws on non-2xx, unifying error handling.
// NOTE: streamChat() below intentionally stays on fetch — axios (XHR) cannot stream
// SSE tokens progressively.
const http = axios.create({ baseURL: API_BASE })

http.interceptors.request.use(async (config) => {
  const auth = await authHeader()
  if (auth.Authorization) config.headers.set('Authorization', auth.Authorization)
  return config
})

/** Stable user id: Clerk subject when authed, else a persistent demo UUID. */
export async function currentUserId(): Promise<string> {
  const clerkId = clerkUserId()
  if (clerkId) return clerkId

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
        // sse-starlette terminates blocks with \r\n\r\n — a literal '\n\n'
        // search never matches, deferring ALL parsing to the post-loop flush
        // (= the whole answer appears in one burst). Match both CRLF and LF.
        let lastBoundary = -1
        let boundaryLen = 0
        const boundaryRe = /\r?\n\r?\n/g
        for (let m = boundaryRe.exec(buffer); m !== null; m = boundaryRe.exec(buffer)) {
          lastBoundary = m.index
          boundaryLen = m[0].length
        }
        if (lastBoundary === -1) continue
        const ready = buffer.slice(0, lastBoundary)
        buffer = buffer.slice(lastBoundary + boundaryLen)
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
  const { data } = await http.get('/sessions', { params: { user_id: userId } })
  return data
}

export async function getSession(sessionId: string) {
  const userId = await currentUserId()
  const { data } = await http.get(`/sessions/${encodeURIComponent(sessionId)}`, {
    params: { user_id: userId },
  })
  return data
}

export async function deleteSession(sessionId: string) {
  const userId = await currentUserId()
  const { data } = await http.delete(
    `/sessions/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`,
  )
  return data
}

// ── On-demand TTS ──────────────────────────────────────────────────────────────

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  tokens?: number
}

/**
 * Speak a message the user asked to hear. Resolves with a playable audio URL.
 *
 * Two hops on purpose. VieNeu is CPU-only at roughly 18ms per character, so a
 * full answer takes 30-45s — far too long to hold a request open. POST /tts
 * returns a task id straight away and the result lands in Redis, which is the
 * same path /chat already uses for its automatic voicing.
 */
export async function speakText(
  text: string,
  opts: { signal?: AbortSignal; pollMs?: number; timeoutMs?: number } = {},
): Promise<string> {
  const { signal, pollMs = 1000, timeoutMs = 180_000 } = opts

  const { data } = await http.post('/tts', { text }, { signal })
  const taskId = (data as { task_id: string }).task_id

  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
    await new Promise((r) => setTimeout(r, pollMs))
    try {
      const res = await http.get(`/tts/${encodeURIComponent(taskId)}/result`, { signal })
      const payload = res.data as { event?: string; url?: string; error?: string }
      if (payload.event === 'speech_ready' && payload.url) return payload.url
      if (payload.event === 'speech_failed') throw new Error(payload.error ?? 'TTS failed')
    } catch (e) {
      // 404 just means "not ready yet" — that is the documented contract of
      // GET /tts/{id}/result, so it must not abort the poll.
      const status = (e as { response?: { status?: number } }).response?.status
      if (status !== 404) throw e
    }
  }
  throw new Error(`TTS timed out after ${Math.round(timeoutMs / 1000)}s`)
}

// ── User memory CRUD ───────────────────────────────────────────────────────────

export async function listUserMemory() {
  const userId = await currentUserId()
  const { data } = await http.get(`/users/${encodeURIComponent(userId)}/memory`)
  return data
}

export async function createUserMemory(factText: string, category?: string) {
  const userId = await currentUserId()
  const { data } = await http.post(`/users/${encodeURIComponent(userId)}/memory`, {
    fact_text: factText,
    category,
  })
  return data
}

export async function deleteUserMemory(factId: string) {
  const userId = await currentUserId()
  const { data } = await http.delete(
    `/users/${encodeURIComponent(userId)}/memory/${encodeURIComponent(factId)}`,
  )
  return data
}

// ── Zero-cost sandbox billing ────────────────────────────────────────────────

export interface BillingConfig {
  sandbox_enabled: boolean
  test_only: true
  real_transactions_enabled: false
  clerk_billing_enabled: false
  stripe_configured: boolean
  checkout_enabled: boolean
  webhook_configured: boolean
}

export interface BillingStatus {
  access_plan: 'FREE' | 'DEMO'
  price: 0
  currency: 'USD'
  stripe_mode: 'test'
  real_transactions_enabled: false
  subscription_status: string | null
  has_test_customer: boolean
}

export async function getBillingConfig(): Promise<BillingConfig> {
  const { data } = await http.get('/billing/config')
  return data as BillingConfig
}

export async function getBillingStatus(): Promise<BillingStatus> {
  const userId = await currentUserId()
  const { data } = await http.get('/billing/status', { params: { user_id: userId } })
  return data as BillingStatus
}

async function openBillingDestination(path: '/billing/checkout' | '/billing/portal') {
  const userId = await currentUserId()
  const { data } = await http.post(path, undefined, { params: { user_id: userId } })
  const url = (data as { url?: string }).url
  if (!url) throw new Error('Stripe sandbox did not return a destination URL')
  window.location.assign(url)
}

export function startSandboxCheckout() {
  return openBillingDestination('/billing/checkout')
}

export function openSandboxPortal() {
  return openBillingDestination('/billing/portal')
}
