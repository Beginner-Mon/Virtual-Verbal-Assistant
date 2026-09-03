/**
 * The pointer to the conversation you are in — never the conversation itself.
 * Postgres stays the source of truth.
 *
 * WHY THIS IS ITS OWN FILE
 * ------------------------
 * It used to be three lines inside ChatContext, and those three lines produced
 * a guaranteed 404 on every clean page load:
 *
 *     localStorage empty
 *       → loadOrCreateSessionId() minted a UUID and stored it immediately
 *       → the mount effect asked GET /sessions/{the id it had just invented}
 *       → the server had never seen it → 404
 *
 * The `conversations` row is written server-side on the first turn
 * (db/session_store.py, INSERT … ON CONFLICT (session_id) DO UPDATE), so the
 * gap between "a pointer exists" and "a row exists" always answered 404. Worse,
 * pressing "new chat" and walking away stored an id that no message ever
 * followed, and that one 404'd on every load from then on — forever. One such
 * id was found in a real browser while the account's seven real conversations
 * were untouched in the database.
 *
 * So the pointer is now written LATE: only when a message is actually sent. No
 * pointer means nothing to restore, which means no request, which means no 404.
 *
 * WHY A TTL, AND WHY IT LIVES HERE
 * --------------------------------
 * localStorage has no expiry — the browser will never drop this key, and
 * nothing in the app used to either, not even on sign-out. A pointer could sit
 * there indefinitely. So the expiry is ours to implement: the timestamp is
 * stored beside the id and the comparison happens on read. Lazy expiry, the way
 * Redis handles most keys — the key can sit for a year and is discarded the
 * moment somebody reads it.
 *
 * Not sessionStorage: that clears when the TAB closes, which is a tab lifetime
 * and not a TTL — closing the browser and coming back ten minutes later would
 * lose your place, which is the exact thing this mechanism exists to prevent.
 * Not a cookie with Max-Age: the browser would expire it for us, but a cookie
 * rides on every request to the origin, and the server never reads this value.
 *
 * The TTL is deliberately client-side. It governs whether a refresh RESUMES a
 * conversation, not whether one can be opened: picking an old conversation out
 * of the sessions list has to keep working however old it is, so
 * GET /sessions/{id} is untouched.
 *
 * `Date.now()` is the user's own clock and they can move it. The worst outcome
 * is resuming a slightly older conversation, or not resuming one. Not defended.
 *
 * `now` and `storage` are parameters so the rules can be tested without a DOM —
 * vitest runs in the `node` environment here.
 */

export const SESSION_KEY = 'vva_session_id'

/** How long a conversation stays "the one you are in" after the last message. */
export const SESSION_TTL_MS = 2 * 60 * 60 * 1000 // 2 hours

interface SessionPointer {
  id: string
  at: number
}

function isPointer(value: unknown): value is SessionPointer {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as SessionPointer).id === 'string' &&
    (value as SessionPointer).id.length > 0 &&
    typeof (value as SessionPointer).at === 'number' &&
    Number.isFinite((value as SessionPointer).at)
  )
}

/** Private mode makes every one of these throw; an unreadable pointer is the
 *  same thing as no pointer. */
function safeStorage(storage?: Storage): Storage | null {
  try {
    return storage ?? window.localStorage
  } catch {
    return null
  }
}

export function clearSessionPointer(storage?: Storage): void {
  const s = safeStorage(storage)
  if (!s) return
  try {
    s.removeItem(SESSION_KEY)
  } catch {
    // Nothing to do about it and nothing depends on it.
  }
}

/**
 * The conversation to resume, or null to start fresh.
 *
 * Drops the key on the way out when it is expired, malformed, or written by the
 * version that stored a bare string. A bare string cannot be dated, and it may
 * well be one of the pointers that never had a row behind it, so it is
 * discarded rather than adopted — the cost is one lost resume, once.
 */
export function readSessionPointer(now: number = Date.now(), storage?: Storage): string | null {
  const s = safeStorage(storage)
  if (!s) return null

  let raw: string | null
  try {
    raw = s.getItem(SESSION_KEY)
  } catch {
    return null
  }
  if (!raw) return null

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    clearSessionPointer(s)
    return null
  }

  if (!isPointer(parsed)) {
    clearSessionPointer(s)
    return null
  }

  if (now - parsed.at > SESSION_TTL_MS) {
    clearSessionPointer(s)
    return null
  }

  return parsed.id
}

/**
 * Point at this conversation, and restart its clock.
 *
 * Called on every send, not only the first, so an active conversation never
 * expires mid-use: the two hours run from the last message rather than from
 * when the conversation began.
 */
export function stampSessionPointer(
  id: string,
  now: number = Date.now(),
  storage?: Storage,
): void {
  const s = safeStorage(storage)
  if (!s) return
  try {
    s.setItem(SESSION_KEY, JSON.stringify({ id, at: now } satisfies SessionPointer))
  } catch {
    // Quota or private mode. The conversation still works for this page load;
    // only the ability to resume it is lost.
  }
}
