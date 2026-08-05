/**
 * Routes a playing TTS clip into the avatar's lip sync.
 *
 * `analyserFromElement` has existed since Phase D but nothing ever called it —
 * the avatar could only mime to the DevPanel's synthetic tone. This is the glue
 * that points it at real speech.
 *
 * Three things here are not optional:
 *
 * 1. `createMediaElementSource` may be called **once** per <audio> element; a
 *    second call throws InvalidStateError. Hence the cache.
 * 2. Routing an element through Web Audio means its sound now leaves via the
 *    graph. If the AudioContext is suspended there is silence — so we refuse to
 *    build the graph unless the context is actually running, and let the caller
 *    play the element plainly instead. Losing lip sync beats losing audio.
 * 3. Cross-origin media (TTS serves from :5000, the app runs on :5173) yields an
 *    all-zero analyser unless the element is CORS-clean. The element must be
 *    created by `createSpeechAudio` below, which sets `crossOrigin` *before* src.
 */

import { analyserFromElement, ensureAudioContext } from '../avatar/lipSyncAudio'

const analysers = new WeakMap<HTMLAudioElement, AnalyserNode>()

/** Incrementing id of whoever is currently driving the mouth. */
let speakerSeq = 0
let activeSpeaker = 0

/**
 * An <audio> element safe to feed into Web Audio.
 *
 * `crossOrigin` has to be set before `src`, which rules out `new Audio(url)`.
 * Note this makes the request CORS-mandatory: a server that sends no
 * `Access-Control-Allow-Origin` will now fail the load outright rather than
 * play without lip sync. The TTS service does send it.
 */
export function createSpeechAudio(src: string): HTMLAudioElement {
  const el = new Audio()
  el.crossOrigin = 'anonymous'
  el.src = src
  return el
}

/** Resume the shared context and report whether it is genuinely running. */
async function runningContext(): Promise<AudioContext | null> {
  const ctx = ensureAudioContext()
  if (ctx.state === 'suspended') {
    try {
      await ctx.resume()
    } catch {
      // Autoplay policy — no recent user gesture. Not an error.
    }
  }
  return ctx.state === 'running' ? ctx : null
}

/**
 * Begin driving the avatar's mouth from this element.
 *
 * @returns a speaker id to pass back to `stopSpeaking`, or 0 when lip sync could
 *          not start (no avatar mounted, context suspended, CORS-tainted audio).
 *          A 0 means the caller should simply play the element as normal audio.
 */
export async function startSpeaking(
  el: HTMLAudioElement,
  controller: { startLipSync: (a: AnalyserNode) => void } | null,
): Promise<number> {
  if (!controller) return 0
  if (!(await runningContext())) return 0

  let analyser = analysers.get(el)
  if (!analyser) {
    try {
      analyser = analyserFromElement(el)
      analysers.set(el, analyser)
    } catch (e) {
      console.warn('[lipsync] could not tap audio element:', e)
      return 0
    }
  }

  activeSpeaker = ++speakerSeq
  controller.startLipSync(analyser)
  return activeSpeaker
}

/**
 * Stop lip sync — but only if this speaker still owns the mouth.
 *
 * Two messages can be played at once. Without this check the first clip to
 * finish would close the mouth of the one still talking.
 */
export function stopSpeaking(
  id: number,
  controller: { stopLipSync: () => void } | null,
): void {
  if (!id || !controller) return
  if (id !== activeSpeaker) return
  activeSpeaker = 0
  controller.stopLipSync()
}
