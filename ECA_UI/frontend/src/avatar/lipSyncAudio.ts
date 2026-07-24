/**
 * Web Audio glue for lip sync. Builds the AnalyserNode that LipSyncController
 * reads. Two sources:
 *   - analyserFromElement: the real path for a `tts.audio` clip (Phase D).
 *   - playSyntheticSpeech: a syllable-like tone for DevPanel testing without a
 *     wav asset (Phase C acceptance).
 *
 * AudioContext must be resumed after a user gesture (autoplay policy, §9); the
 * DevPanel button click / send action satisfies that.
 */

let sharedCtx: AudioContext | null = null

export function ensureAudioContext(): AudioContext {
  if (!sharedCtx) {
    const Ctor: typeof AudioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    sharedCtx = new Ctor()
  }
  if (sharedCtx.state === 'suspended') void sharedCtx.resume()
  return sharedCtx
}

const FFT_SIZE = 1024

/** Build an analyser fed by an <audio> element. Returns the analyser; caller plays the element. */
export function analyserFromElement(el: HTMLAudioElement): AnalyserNode {
  const ctx = ensureAudioContext()
  const source = ctx.createMediaElementSource(el)
  const analyser = ctx.createAnalyser()
  analyser.fftSize = FFT_SIZE
  source.connect(analyser)
  analyser.connect(ctx.destination)
  return analyser
}

export interface SyntheticSpeech {
  analyser: AnalyserNode
  stop: () => void
}

/**
 * Play a syllable-modulated tone for ~durationMs and return its analyser.
 * A low-frequency oscillator swings the gain so RMS rises and falls like speech,
 * exercising the full RMS -> viseme -> mouth path with no audio file.
 */
export function playSyntheticSpeech(durationMs = 3000): SyntheticSpeech {
  const ctx = ensureAudioContext()

  const voice = ctx.createOscillator()
  voice.type = 'sawtooth'
  voice.frequency.value = 140

  const gain = ctx.createGain()
  gain.gain.value = 0.35

  // LFO ~4.5Hz modulates the gain to mimic syllables.
  const lfo = ctx.createOscillator()
  lfo.type = 'sine'
  lfo.frequency.value = 4.5
  const lfoDepth = ctx.createGain()
  lfoDepth.gain.value = 0.3
  lfo.connect(lfoDepth)
  lfoDepth.connect(gain.gain)

  const analyser = ctx.createAnalyser()
  analyser.fftSize = FFT_SIZE

  voice.connect(gain)
  gain.connect(analyser)
  analyser.connect(ctx.destination)

  const now = ctx.currentTime
  voice.start(now)
  lfo.start(now)
  const stopAt = now + durationMs / 1000
  voice.stop(stopAt)
  lfo.stop(stopAt)

  const stop = () => {
    try {
      voice.stop()
      lfo.stop()
    } catch {
      /* already stopped */
    }
  }

  return { analyser, stop }
}
