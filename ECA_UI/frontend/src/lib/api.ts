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

import axios from 'axios'
import { fetchAuthSession } from 'aws-amplify/auth'

// ── Config ────────────────────────────────────────────────────────────────────

const DEFAULT_API_BASE = 'http://localhost:8000'

/**
 * Missing configuration warns; it does not throw.
 *
 * This used to `throw` at module load. Vite bakes env vars in at BUILD time and
 * `.env.local` is gitignored, so a CI build has none — and a throw at import
 * time takes down the whole app before it renders a pixel. The deployed page was
 * a blank screen whose only symptom was a message telling you to create a file
 * on a machine that is not the one serving it.
 *
 * The value is only needed when a request is actually made, so a bad one costs a
 * failed request, not a dead app. That trade is right in both directions: the UI
 * can be reviewed without a backend, and the console says exactly what is wrong.
 *
 * Note the port: 8080 is taken by another service on the dev machine — the
 * backend is on 8000. The old message named 8080 while the comment above it said
 * 8000.
 */
const _raw = import.meta.env.VITE_API_BASE_URL as string | undefined
if (!_raw) {
  console.warn(
    `[api] VITE_API_BASE_URL is not set — falling back to ${DEFAULT_API_BASE}. ` +
      'Vite reads this at BUILD time: set it in .env.local for local dev, or as ' +
      'an Amplify Console environment variable for a deployed build. Requests ' +
      'will fail until it points at a reachable backend.'
  )
}
const API_BASE: string = _raw || DEFAULT_API_BASE

/**
 * Where the session and user-memory endpoints live.
 *
 * Those moved to their own Lambda: serving a session list should not require
 * keeping the whole agent process alive, since that one imports torch at boot to
 * run a few SQL statements. /chat and /tts stay on API_BASE because they need
 * the graph.
 *
 * Defaults to API_BASE, so an unset value means "one backend, as before" rather
 * than a broken app — the same reasoning as the warning above. There is a
 * precedent for the split: VITE_ASSET_BASE_URL in lib/characters.ts already
 * points the character catalog at CloudFront.
 */
const CRUD_BASE: string =
  (import.meta.env.VITE_CRUD_API_URL as string | undefined) || API_BASE

// ── Auth helpers ──────────────────────────────────────────────────────────────

/**
 * Returns { Authorization: 'Bearer <idToken>' } when signed in, else {}.
 *
 * Reads the session from Amplify rather than a bridge object the UI has to
 * register. `ea57480` routed this through Clerk instead, and the registration
 * only ever happened inside a Clerk-specific guard that the app never rendered —
 * so this returned {} for every request and the API was called with no
 * credentials at all. Nothing failed loudly; requests just went out anonymous.
 */
async function authHeader(): Promise<Record<string, string>> {
  try {
    const session = await fetchAuthSession()
    const token = session.tokens?.idToken?.toString()
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

// ── Axios instance (REST only) ──────────────────────────────────────────────────
// Used for non-streaming CRUD (sessions, user_memory). The request interceptor
// attaches the Cognito idToken once, so callers don't repeat auth wiring; axios
// also throws on non-2xx, unifying error handling.
// NOTE: streamChat() below intentionally stays on fetch — axios (XHR) cannot stream
// SSE tokens progressively.
/**
 * One factory, so the two instances cannot drift apart on auth.
 *
 * The backend takes identity only from this header — there is no user_id
 * parameter to fall back on any more — so an instance created without the
 * interceptor does not degrade, it 401s on every call.
 */
function makeClient(baseURL: string) {
  const client = axios.create({ baseURL })
  client.interceptors.request.use(async (config) => {
    const auth = await authHeader()
    if (auth.Authorization) config.headers.set('Authorization', auth.Authorization)
    return config
  })
  return client
}

// Agent: /tts and its result polling. /chat uses fetch, see streamChat.
const http = makeClient(API_BASE)

// CRUD Lambda: sessions, user memory, billing.
const crud = makeClient(CRUD_BASE)

/**
 * Fire-and-forget wake-up, called once when the app mounts.
 *
 * Both halves of the first request are cold otherwise: the Lambda container and
 * Neon's compute, which suspends when idle. Doing it at mount spends that time
 * while the user is still looking at the shell, instead of on their first click.
 * Failures are ignored on purpose — this is an optimisation, and the real
 * request will report anything genuinely wrong.
 */
export function wakeCrudApi(): void {
  void fetch(`${CRUD_BASE}/health`, { method: 'GET' }).catch(() => {})
}

/**
 * The signed-in user's Cognito sub, else a per-browser demo UUID.
 *
 * NOT an identity as far as the API is concerned, and no longer sent to it.
 * Every endpoint now derives the user from the verified token; the demo branch
 * below reaches a server that answers 401, whatever it returns. Treat this as a
 * client-side key — for local storage, cache keys, UI state — and nothing more.
 *
 * The branch is kept because the UI still needs *something* stable to key on
 * before sign-in. If you find yourself passing the result to the API, that is
 * the bug this comment exists to prevent.
 */
export async function currentUserId(): Promise<string> {
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

  const extraHeaders = await authHeader()

  const resp = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...extraHeaders,
    },
    // No user_id: the server reads identity from the Authorization header, and
    // ChatRequest no longer has the field.
    body: JSON.stringify({
      query,
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

// No user_id anywhere below. The server reads it from the Bearer token, so
// sending one would be a claim about who we are rather than a parameter — and
// the routes no longer accept it.

export async function listSessions() {
  const { data } = await crud.get('/sessions')
  return data
}

export async function getSession(sessionId: string) {
  const { data } = await crud.get(`/sessions/${encodeURIComponent(sessionId)}`)
  return data
}

export async function deleteSession(sessionId: string) {
  const { data } = await crud.delete(`/sessions/${encodeURIComponent(sessionId)}`)
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
  const { data } = await crud.get('/me/memory')
  return data
}

export async function createUserMemory(factText: string, category?: string) {
  const { data } = await crud.post('/me/memory', {
    fact_text: factText,
    category,
  })
  return data
}

export async function deleteUserMemory(factId: string) {
  const { data } = await crud.delete(`/me/memory/${encodeURIComponent(factId)}`)
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

// Billing stays on API_BASE: the feature is unfinished, so it was left out of
// the CRUD Lambda move. It has dropped its user_id parameter all the same —
// that was the auth change, which applies to every endpoint regardless of where
// it is served.

export async function getBillingConfig(): Promise<BillingConfig> {
  const { data } = await http.get('/billing/config')
  return data as BillingConfig
}

export async function getBillingStatus(): Promise<BillingStatus> {
  const { data } = await http.get('/billing/status')
  return data as BillingStatus
}

async function openBillingDestination(path: '/billing/checkout' | '/billing/portal') {
  const { data } = await http.post(path, undefined)
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
