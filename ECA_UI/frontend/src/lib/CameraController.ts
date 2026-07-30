/**
 * Camera framing, driven by FSM state (plan §3.4).
 *
 * Owns only the *mode* (`head` close-up vs `hips` full-body) and the cooldown
 * timer. The orbit/follow maths stays in the R3F scene, which needs the live
 * camera and controls.
 *
 * Key rule: this timer changes the CAMERA only — it must never call
 * `transitionTo`. A system may have exactly one thing driving state. Plan v1.0
 * let a camera timer own an FSM state (`exercise_cooldown`) and that produced a
 * permanent deadlock (bug 🔴 #1).
 */

import { cameraModeOf, type CameraMode, type CharState } from './AnimationStates'

/** How long the wide framing is held after LEAVING a wide state. */
const COOLDOWN_MS = 3000

export class CameraController {
  private state: CharState = 'idle'
  private mode: CameraMode = 'head'
  private timer: ReturnType<typeof setTimeout> | null = null

  private readonly onModeChanged: (mode: CameraMode) => void

  constructor(onModeChanged: (mode: CameraMode) => void) {
    this.onModeChanged = onModeChanged
  }

  get cameraMode(): CameraMode {
    return this.mode
  }

  onStateChanged(next: CharState): void {
    const wasWide = cameraModeOf(this.state) === 'hips'
    const isWide = cameraModeOf(next) === 'hips'
    this.state = next
    this.clearTimer()

    if (isWide) {
      this.set('hips')
      return
    }
    if (wasWide) {
      // Keep the wide shot a moment longer so the motion's last pose is visible,
      // then ease back to the face. The FSM has already moved on to `idle`.
      this.timer = setTimeout(() => {
        this.timer = null
        this.set('head')
      }, COOLDOWN_MS)
      return
    }
    this.set('head')
  }

  /** Manual override from the debug panel. Cancels any pending cooldown. */
  setMode(mode: CameraMode): void {
    this.clearTimer()
    this.set(mode)
  }

  dispose(): void {
    this.clearTimer()
  }

  private set(mode: CameraMode): void {
    if (this.mode === mode) return
    this.mode = mode
    this.onModeChanged(mode)
  }

  private clearTimer(): void {
    if (this.timer === null) return
    clearTimeout(this.timer)
    this.timer = null
  }
}
