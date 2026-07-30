/**
 * Animation FSM runtime: the only thing that changes character state.
 *
 * Knows about `AnimationClip`s, actions and cross-fades. Knows NOTHING about
 * URLs, file formats, retargeting or subclips — that is `AnimationRegistry`
 * (plan §3.3).
 *
 * CRITICAL INVARIANT — never render the bind pose (T-pose):
 *
 *     load new clip → play() new action → fade out old
 *
 * never `stop old → load → play new`, because the load is async and the gap
 * would show the skeleton's bind pose. This invariant already holds in the
 * pre-refactor code; it is preserved here, not introduced.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import type { AnimationRegistry } from './AnimationRegistry'
import { canTransition, crossfadeFor, loopModeOf, onFinishedOf, type CharState } from './AnimationStates'

export interface ClipInfo {
  state: CharState
  tracks: number
  duration: number
}

export interface AnimationControllerOptions {
  /**
   * Fires when a clip has actually been applied to the skeleton. This is the
   * readiness signal the loading overlay waits on — an event, never a timeout.
   */
  onClipApplied?: (info: ClipInfo) => void
}

type StateListener = (state: CharState) => void
type EventName = 'stateChanged' | 'finished'

/** Weight below which a fading-out action is considered done and retired. */
const RETIRE_WEIGHT = 0.001

export class AnimationController {
  private readonly mixer: THREE.AnimationMixer
  private state: CharState = 'idle'
  private currentAction: THREE.AnimationAction | null = null
  private fading: THREE.AnimationAction[] = []
  /** Cancels stale transitions when several are fired in quick succession. */
  private loadGen = 0
  private playing = true
  private disposed = false
  private readonly listeners: Record<EventName, Set<StateListener>> = {
    stateChanged: new Set(),
    finished: new Set(),
  }

  private readonly vrm: VRM
  private readonly registry: AnimationRegistry
  private readonly options: AnimationControllerOptions

  constructor(
    vrm: VRM,
    registry: AnimationRegistry,
    options: AnimationControllerOptions = {},
  ) {
    this.vrm = vrm
    this.registry = registry
    this.options = options
    this.mixer = new THREE.AnimationMixer(vrm.scene)
    this.mixer.addEventListener('finished', this.handleFinished)
  }

  /**
   * Current state. Note this is `'idle'` before the first successful
   * transition, while `currentAction` is still null — that is what makes the
   * boot transition to `greeting` legal from frame zero (plan §2.5).
   */
  get currentState(): CharState {
    return this.state
  }

  get hasPose(): boolean {
    return this.currentAction !== null
  }

  /**
   * Single entry point for state changes. Returns false when the transition is
   * not allowed, the clip is unavailable, or a newer transition superseded it.
   * Callers that must not end up poseless (boot) have to handle false.
   */
  async transitionTo(next: CharState): Promise<boolean> {
    if (this.disposed) return false
    if (!canTransition(this.state, next)) return false

    // Re-entering a looping state is a no-op: restarting it would visibly jump.
    // One-shots DO restart (exercise → exercise means "play the new motion").
    if (next === this.state && this.currentAction && loopModeOf(next) === 'repeat') return true

    const gen = ++this.loadGen
    const clip = await this.registry.get(next)
    if (!clip || gen !== this.loadGen || this.disposed) return false

    const from = this.state
    const once = loopModeOf(next) === 'once'
    const action = this.mixer.clipAction(clip)
    action.setLoop(once ? THREE.LoopOnce : THREE.LoopRepeat, once ? 1 : Infinity)
    action.clampWhenFinished = once

    // `clipAction` returns the SAME action instance for a given clip. A one-shot
    // that already ran is parked on its last frame with `paused = true`, and
    // `play()` does not rewind — without reset() a second `exercise` (or
    // `greeting`) would simply not move.
    action.reset()

    const prev = this.currentAction

    if (!this.playing) {
      // Debug pause: adopt the state without animating. `setPlaying(true)` will
      // start this action from the top.
      if (prev && prev !== action) prev.stop()
      this.commit(next, action, clip)
      return true
    }

    action.play()

    if (prev && prev !== action) {
      const fade = crossfadeFor(from, next)
      // A finished LoopOnce action sits at weight 0 with clampWhenFinished;
      // restoring weight first makes the fade-out actually visible.
      if (!prev.isRunning()) prev.setEffectiveWeight(1)
      prev.fadeOut(fade)
      action.fadeIn(fade)
      this.fading = this.fading.filter((a) => a !== action)
      this.fading.push(prev)
    } else if (!prev) {
      // First action on this model: full weight immediately. Fading in from zero
      // would expose the bind pose for the whole fade duration.
      action.setEffectiveWeight(1)
    }

    this.commit(next, action, clip)
    return true
  }

  /** Per-frame tick. Call from R3F's `useFrame`, before the facial controller. */
  update(delta: number): void {
    if (this.disposed) return
    this.mixer.update(delta)

    for (let i = this.fading.length - 1; i >= 0; i--) {
      const action = this.fading[i]
      if (action === this.currentAction || action.getEffectiveWeight() > RETIRE_WEIGHT) continue
      action.stop()
      this.mixer.uncacheAction(action.getClip())
      this.fading.splice(i, 1)
    }
  }

  /**
   * Debug-only playback toggle. Pausing STOPS the action, which restores the
   * bind pose — acceptable behind a dev control, never on a user path.
   */
  setPlaying(playing: boolean): void {
    this.playing = playing
    const action = this.currentAction
    if (!action) return
    if (playing) {
      if (!action.isRunning()) {
        action.reset()
        action.play()
        action.setEffectiveWeight(1)
      } else {
        action.paused = false
      }
    } else {
      action.stop()
    }
  }

  setSpeed(speed: number): void {
    this.mixer.timeScale = speed
  }

  /** Restart the current clip from frame 0. */
  restart(): void {
    const action = this.currentAction
    if (!action) return
    action.reset()
    if (this.playing) {
      action.play()
      action.setEffectiveWeight(1)
    }
  }

  on(event: EventName, listener: StateListener): () => void {
    this.listeners[event].add(listener)
    return () => this.listeners[event].delete(listener)
  }

  dispose(): void {
    this.disposed = true
    this.mixer.removeEventListener('finished', this.handleFinished)
    this.mixer.stopAllAction()
    this.mixer.uncacheRoot(this.vrm.scene)
    this.currentAction = null
    this.fading = []
    this.listeners.stateChanged.clear()
    this.listeners.finished.clear()
  }

  private commit(next: CharState, action: THREE.AnimationAction, clip: THREE.AnimationClip): void {
    const isFirstPose = this.currentAction === null
    this.currentAction = action
    this.state = next
    // Flush the very first pose onto the bones before anyone is told we are
    // ready — the reveal gate must never unhide a skeleton still in bind pose.
    if (isFirstPose) this.mixer.update(0)
    this.options.onClipApplied?.({
      state: next,
      tracks: clip.tracks.length,
      duration: clip.duration,
    })
    this.emit('stateChanged', next)
  }

  /**
   * A one-shot ended. Advancing is done HERE rather than by an external
   * listener: `onFinished` is part of the state definition, so leaving the hop
   * to a caller would mean every call site could forget it and strand the FSM
   * on a clip's last frame. Observers still get the `finished` event.
   */
  private handleFinished = (event: { action: THREE.AnimationAction }): void => {
    if (event.action !== this.currentAction || this.disposed) return
    const completed = this.state
    this.emit('finished', completed)
    const next = onFinishedOf(completed)
    if (next) void this.transitionTo(next)
  }

  private emit(event: EventName, state: CharState): void {
    this.listeners[event].forEach((listener) => listener(state))
  }
}
