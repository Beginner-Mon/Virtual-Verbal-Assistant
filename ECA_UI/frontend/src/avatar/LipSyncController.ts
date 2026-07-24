import type { ExpressionContributor } from './ExpressionMixer'
import type { AvatarProfile } from './AvatarProfile'

/**
 * Amplitude-based lip sync — Mode 1 only (facial-animation-plan.md §9).
 * VieNeu-TTS-GGUF exports no phoneme timestamps, so Mode 2 (phoneme visemes) is
 * left as a future interface, not built.
 *
 * Owns the mouth viseme channels. Runs AFTER the emotion contributor in the
 * mixer so it OVERRIDES the mouth while audio plays; when silent it decays to 0
 * and emotion owns the mouth again (§5).
 *
 * Reads RMS from an AnalyserNode each frame -> smoothed open weight (fast attack,
 * slower release). The audio clock is the source of truth; wall-clock is not used.
 */
const RMS_GAIN = 5
const ATTACK_PER_SEC = 40
const RELEASE_PER_SEC = 12
const MIN_VISIBLE = 0.01

export class LipSyncController implements ExpressionContributor {
  private readonly aaChannel: string
  private readonly ouChannel: string

  private analyser: AnalyserNode | null = null
  private buffer: Uint8Array = new Uint8Array(0)
  private active = false
  private weight = 0

  constructor(profile: AvatarProfile) {
    this.aaChannel = profile.visemes.A
    this.ouChannel = profile.visemes.U
  }

  /** Begin driving the mouth from this analyser (created by the audio glue). */
  start(analyser: AnalyserNode): void {
    this.analyser = analyser
    this.buffer = new Uint8Array(analyser.fftSize)
    this.active = true
  }

  /** Stop reading; the mouth decays closed over the release time. */
  stop(): void {
    this.active = false
    this.analyser = null
  }

  get isPlaying(): boolean {
    return this.active
  }

  tick(delta: number): void {
    let target = 0
    if (this.active && this.analyser) {
      this.analyser.getByteTimeDomainData(this.buffer)
      let sumSq = 0
      for (let i = 0; i < this.buffer.length; i++) {
        const v = (this.buffer[i] - 128) / 128
        sumSq += v * v
      }
      const rms = Math.sqrt(sumSq / this.buffer.length)
      target = clamp01(rms * RMS_GAIN)
    }

    // Asymmetric, frame-rate-independent smoothing: snap open, ease closed.
    const rate = target > this.weight ? ATTACK_PER_SEC : RELEASE_PER_SEC
    const k = 1 - Math.exp(-rate * delta)
    this.weight += (target - this.weight) * k
  }

  contribute(frame: Map<string, number>): void {
    if (this.weight <= MIN_VISIBLE) return
    // Open jaw (aa) with a touch of rounding (ou) for a less robotic shape.
    frame.set(this.aaChannel, this.weight)
    frame.set(this.ouChannel, this.weight * 0.35)
  }

  /** Debug read (verification only). */
  debugWeight(): number {
    return Number(this.weight.toFixed(3))
  }

  detach(): void {
    this.stop()
    this.weight = 0
  }
}

function clamp01(v: number): number {
  if (v < 0) return 0
  if (v > 1) return 1
  return v
}
