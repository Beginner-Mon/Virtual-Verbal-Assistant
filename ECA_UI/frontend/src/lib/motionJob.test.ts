import { describe, expect, it, vi } from 'vitest'

import { pollMotionJob, type MotionStatus } from './motionJob'

/** Zero interval so the loop runs at test speed; the real caller passes ~1500ms. */
const FAST = { pollMs: 0, timeoutMs: 5_000 }

function fetcherFor(...responses: MotionStatus[]) {
  const fn = vi.fn<(jobId: string) => Promise<MotionStatus>>()
  for (const r of responses) fn.mockResolvedValueOnce(r)
  // Anything after the scripted responses keeps returning the last one, so a
  // test that over-polls hangs on a stable value instead of throwing a
  // confusing "undefined" from an exhausted mock.
  fn.mockResolvedValue(responses[responses.length - 1])
  return fn
}

describe('pollMotionJob', () => {
  it('resolves with the signed URL once the render is done', async () => {
    const fetch = fetcherFor(
      { status: 'queued' },
      { status: 'processing' },
      { status: 'done', url: 'https://cdn/motions/abc.bvh?Signature=s' },
    )
    await expect(pollMotionJob('abc', fetch, FAST)).resolves.toBe(
      'https://cdn/motions/abc.bvh?Signature=s',
    )
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it('stops on failed instead of spinning to the timeout', async () => {
    // The whole point of a terminal state. Treating `failed` as "not ready yet"
    // would keep an avatar spinner up for the full timeout for a render that is
    // never coming.
    const fetch = fetcherFor({ status: 'failed', reason: 'lease expired' })
    await expect(pollMotionJob('abc', fetch, FAST)).rejects.toThrow('lease expired')
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('stops on not_found', async () => {
    // GET /motion/{id} answers 404 for a row that never existed or has aged out
    // (DynamoDB TTL is 24h). api.ts maps that 404 to this status.
    //
    // NOTE THE INVERSION: on /tts/{id}/result a 404 means "not ready, keep
    // polling". Here it is terminal. Copying the TTS loop shape without
    // changing this is a loop that never ends.
    const fetch = fetcherFor({ status: 'not_found' })
    await expect(pollMotionJob('gone', fetch, FAST)).rejects.toThrow(/not_found|not found/i)
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('reports done-without-a-url as a failure rather than resolving undefined', async () => {
    const fetch = fetcherFor({ status: 'done' })
    await expect(pollMotionJob('abc', fetch, FAST)).rejects.toThrow()
  })

  it('gives up at the deadline', async () => {
    const fetch = fetcherFor({ status: 'queued' })
    await expect(
      pollMotionJob('abc', fetch, { pollMs: 0, timeoutMs: 30 }),
    ).rejects.toThrow(/timed out/i)
  })

  it('aborts when the signal fires, without another fetch', async () => {
    // ChatContext hands this the same AbortController as the chat stream, so
    // switching session or sending a new message must stop the poll.
    const controller = new AbortController()
    const fetch = vi.fn<(jobId: string) => Promise<MotionStatus>>(async () => {
      controller.abort()
      return { status: 'queued' }
    })
    await expect(
      pollMotionJob('abc', fetch, { ...FAST, signal: controller.signal }),
    ).rejects.toThrow(/abort/i)
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})
