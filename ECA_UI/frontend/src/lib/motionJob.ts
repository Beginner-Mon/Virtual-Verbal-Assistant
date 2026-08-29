/**
 * Poll a motion render job until it produces a playable URL.
 *
 * The GPU renders a BVH clip in about three seconds, but the request that
 * starts it returns immediately — the agent writes a row to DynamoDB and a
 * worker picks it up out of band. So the browser learns the job id from the
 * `motion` SSE event and then asks `GET /motion/{job_id}` until there is
 * something to play.
 *
 * The fetch is injected rather than imported. api.ts builds its axios client at
 * module scope, which pulls in aws-amplify for the JWT interceptor; keeping the
 * loop free of that import is what lets this file be tested under vitest's
 * `environment: 'node'` with no DOM and no auth.
 *
 * ── The one thing to get right ────────────────────────────────────────────
 * This looks like `speakText`, and the 404 means the opposite:
 *
 *     GET /tts/{id}/result   404 → not ready yet, keep polling
 *     GET /motion/{job_id}   404 → the row is gone. STOP.
 *
 * api.ts maps that 404 to `{status: 'not_found'}`, which is terminal here.
 * Copying the TTS loop's shape without inverting this gives a loop that runs to
 * its timeout for a job that will never arrive.
 */

export interface MotionStatus {
  /** done | queued | processing | failed | not_found */
  status: string
  /** CloudFront signed URL — present only on `done`, and valid for 5 minutes. */
  url?: string
  reason?: string
}

export type MotionFetcher = (jobId: string) => Promise<MotionStatus>

/** Terminal states. Anything else means "still working, ask again". */
const DONE = 'done'
const TERMINAL_FAILURES = new Set(['failed', 'not_found'])

export async function pollMotionJob(
  jobId: string,
  fetchStatus: MotionFetcher,
  opts: { signal?: AbortSignal; pollMs?: number; timeoutMs?: number } = {},
): Promise<string> {
  // 1500ms against a ~3s render: two or three requests for a fresh job.
  const { signal, pollMs = 1500, timeoutMs = 120_000 } = opts
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')

    // Fetch BEFORE sleeping, unlike speakText. A `cache_hit` is already
    // rendered — somebody else asked for this exact movement — and sleeping
    // first would put a needless 1.5s in front of an instant answer.
    const res = await fetchStatus(jobId)

    if (res.status === DONE) {
      if (!res.url) {
        // The route reports this itself as `failed`, so reaching here means the
        // shape changed. Fail loudly rather than resolving `undefined` into the
        // animation loader.
        throw new Error('motion job reported done with no URL')
      }
      return res.url
    }
    if (TERMINAL_FAILURES.has(res.status)) {
      throw new Error(res.reason ?? res.status)
    }

    await new Promise((r) => setTimeout(r, pollMs))
  }

  throw new Error(`motion job timed out after ${Math.round(timeoutMs / 1000)}s`)
}
