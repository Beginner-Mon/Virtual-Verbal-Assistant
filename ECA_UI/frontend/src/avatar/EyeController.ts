import type { VRM } from '@pixiv/three-vrm'

/**
 * Drives eye gaze via vrm.lookAt (facial-animation-plan.md §3, §5 look channels
 * are reserved to lookAt — the expression mixer never touches them).
 *
 * Input is normalized [-1..1] mouse coords OR a wander target (set by the idle
 * controller). Mouse wins while it was moved recently; otherwise the wander /
 * center target takes over. Gaze is smoothed frame-rate-independently.
 *
 * We set yaw/pitch (degrees) directly with autoUpdate off: the setters flag the
 * lookAt dirty and vrm.update() applies them through whichever applier the model
 * uses (bone- or expression-based), so this works on every VRM.
 *
 * three-vrm pitch convention (verified in three-vrm-core 3.5.3): NEGATIVE pitch
 * looks UP, positive looks down — VRMLookAt.lookAt() computes
 * `pitch = altitudeFrom - altitudeTo`, so a target above the head yields a
 * negative pitch. Screen-space input is y-up, so every pitch below negates ny.
 */
const YAW_MAX_DEG = 22
const PITCH_MAX_DEG = 12
const MOUSE_ACTIVE_MS = 2000
const SMOOTH_PER_SEC = 10 // higher = snappier

export class EyeController {
  private readonly vrm: VRM

  private curYaw = 0
  private curPitch = 0
  private mouseYaw = 0
  private mousePitch = 0
  private wanderYaw = 0
  private wanderPitch = 0
  private lastMouseAt = -Infinity

  constructor(vrm: VRM) {
    this.vrm = vrm
    if (vrm.lookAt) {
      // Take manual control; we push yaw/pitch each frame.
      vrm.lookAt.autoUpdate = false
      vrm.lookAt.target = null
    }
  }

  /** Feed normalized mouse position in [-1..1] (x right+, y up+). */
  setMouse(nx: number, ny: number, now: number): void {
    this.mouseYaw = clampUnit(nx) * YAW_MAX_DEG
    this.mousePitch = -clampUnit(ny) * PITCH_MAX_DEG // negate: three-vrm pitch+ = down
    this.lastMouseAt = now
  }

  /** Idle gaze wander target in [-1..1] (set by IdleBehaviorController). */
  setWander(nx: number, ny: number): void {
    this.wanderYaw = clampUnit(nx) * YAW_MAX_DEG
    this.wanderPitch = -clampUnit(ny) * PITCH_MAX_DEG // negate: three-vrm pitch+ = down
  }

  tick(delta: number, now: number): void {
    const mouseActive = now - this.lastMouseAt < MOUSE_ACTIVE_MS
    const targetYaw = mouseActive ? this.mouseYaw : this.wanderYaw
    const targetPitch = mouseActive ? this.mousePitch : this.wanderPitch

    // Exponential smoothing, frame-rate independent.
    const k = 1 - Math.exp(-SMOOTH_PER_SEC * delta)
    this.curYaw += (targetYaw - this.curYaw) * k
    this.curPitch += (targetPitch - this.curPitch) * k

    const lookAt = this.vrm.lookAt
    if (lookAt) {
      lookAt.yaw = this.curYaw
      lookAt.pitch = this.curPitch
    }
  }

  /**
   * Precise smoothed gaze angles (degrees), three-vrm convention (yaw+ = right,
   * pitch- = up). HeadController reads these so the head follows the SAME
   * direction as the eyes. Not rounded — unlike debugAngles() — so head
   * smoothing keeps full precision.
   */
  get currentYaw(): number {
    return this.curYaw
  }

  get currentPitch(): number {
    return this.curPitch
  }

  /** Debug read (verification only). */
  debugAngles(): { yaw: number; pitch: number } {
    return { yaw: Number(this.curYaw.toFixed(2)), pitch: Number(this.curPitch.toFixed(2)) }
  }

  detach(): void {
    const lookAt = this.vrm.lookAt
    if (lookAt) {
      lookAt.yaw = 0
      lookAt.pitch = 0
      lookAt.autoUpdate = true
    }
  }
}

function clampUnit(v: number): number {
  if (v < -1) return -1
  if (v > 1) return 1
  return v
}
