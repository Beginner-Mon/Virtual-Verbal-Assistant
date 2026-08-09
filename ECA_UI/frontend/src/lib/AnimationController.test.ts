import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import type { AnimationRegistry } from './AnimationRegistry'
import { AnimationController, type ClipInfo } from './AnimationController'
import type { CharState } from './AnimationStates'

/**
 * A scene with one animatable node, and clips whose tracks actually target it.
 * A clip with no bindable tracks makes the mixer treat every action as inert,
 * so weights stay at their defaults and the cross-fade assertions below would
 * pass without the code doing anything.
 */
function makeVrm(): VRM {
  const scene = new THREE.Group()
  const joint = new THREE.Object3D()
  joint.name = 'Joint'
  scene.add(joint)
  return { scene } as unknown as VRM
}

function makeClip(name: string, duration = 1) {
  return new THREE.AnimationClip(name, duration, [
    new THREE.VectorKeyframeTrack('Joint.position', [0, duration], [0, 0, 0, 0, 1, 0]),
  ])
}

/**
 * Registry double. `get` is async on purpose — every ordering guarantee in the
 * controller (no bind pose, stale-transition cancellation) only exists because
 * clip loading is asynchronous, so a synchronous fake would test nothing.
 */
function makeRegistry(overrides: Partial<Record<CharState, THREE.AnimationClip | null>> = {}) {
  const clips = new Map<CharState, THREE.AnimationClip | null>()
  let pending: (() => void)[] = []
  let gate = false

  const registry = {
    get: vi.fn(async (state: CharState) => {
      if (state in overrides) return overrides[state] ?? null
      if (!clips.has(state)) clips.set(state, makeClip(state))
      const clip = clips.get(state)!
      if (gate) await new Promise<void>((resolve) => pending.push(resolve))
      return clip
    }),
  } as unknown as AnimationRegistry

  return {
    registry,
    clipFor: (state: CharState) => clips.get(state),
    /** Make subsequent `get` calls hang until `release()`. */
    hold: () => { gate = true },
    release: () => {
      gate = false
      pending.forEach((resolve) => resolve())
      pending = []
    },
  }
}

let vrm: VRM

beforeEach(() => {
  vrm = makeVrm()
})

describe('AnimationController — transitions', () => {
  it('starts in idle with no pose', () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    expect(controller.currentState).toBe('idle')
    expect(controller.hasPose).toBe(false)
  })

  it('boots into greeting from the poseless idle state', () => {
    // `greeting` is reach: 'from-idle', and at boot `state` is already 'idle'
    // with a null action. If this were blocked the character would never play
    // its entrance.
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    return controller.transitionTo('greeting').then((ok) => {
      expect(ok).toBe(true)
      expect(controller.currentState).toBe('greeting')
      expect(controller.hasPose).toBe(true)
    })
  })

  it('refuses a transition the FSM disallows', async () => {
    // thinking_outro is reach: { after: ['thinking_intro'] }.
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    expect(await controller.transitionTo('thinking_outro')).toBe(false)
    expect(controller.currentState).toBe('idle')
    expect(registry.get).not.toHaveBeenCalled()
  })

  it('returns false when the clip is unavailable, without changing state', async () => {
    const { registry } = makeRegistry({ greeting: null })
    const controller = new AnimationController(vrm, registry)

    expect(await controller.transitionTo('greeting')).toBe(false)
    expect(controller.currentState).toBe('idle')
    expect(controller.hasPose).toBe(false)
  })

  it('treats re-entering a looping state as a no-op', async () => {
    // Restarting a loop would visibly snap the character back to frame 0.
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    await controller.transitionTo('idle')

    controller.update(0.4)
    const before = (registry.get as ReturnType<typeof vi.fn>).mock.calls.length

    expect(await controller.transitionTo('idle')).toBe(true)
    expect((registry.get as ReturnType<typeof vi.fn>).mock.calls.length).toBe(before)
  })

  it('restarts a one-shot when it is re-entered', async () => {
    // `mixer.clipAction` returns the SAME action for a clip, and a finished
    // one-shot is parked on its last frame with paused = true. Without the
    // reset() in transitionTo, a second `exercise` would simply not move.
    const { registry, clipFor } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    await controller.transitionTo('exercise')
    controller.update(0.9)

    const mixer = (controller as unknown as { mixer: THREE.AnimationMixer }).mixer
    const action = mixer.clipAction(clipFor('exercise')!)
    expect(action.time).toBeGreaterThan(0)

    await controller.transitionTo('exercise')
    expect(action.time).toBe(0)
    expect(action.paused).toBe(false)
  })
})

describe('AnimationController — the no-bind-pose invariant', () => {
  it('holds the old pose for the whole duration of an async load', async () => {
    // The invariant from the file header: load → play new → fade out old. If
    // the controller stopped the old action first, the skeleton would show its
    // bind pose (T-pose) for as long as the load took.
    const { registry, hold, release } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    await controller.transitionTo('idle')

    hold()
    const pending = controller.transitionTo('exercise')

    expect(controller.hasPose).toBe(true)
    expect(controller.currentState).toBe('idle')

    release()
    await pending
    expect(controller.currentState).toBe('exercise')
  })

  it('gives the very first action full weight instead of fading it in', async () => {
    // Fading in from zero would expose the bind pose for the whole fade.
    const { registry, clipFor } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    await controller.transitionTo('idle')

    const mixer = (controller as unknown as { mixer: THREE.AnimationMixer }).mixer
    expect(mixer.clipAction(clipFor('idle')!).getEffectiveWeight()).toBe(1)
  })

  it('reports the clip as applied only once it is on the bones', async () => {
    // This event is what the loading overlay waits on — an event, never a
    // timeout. It must fire after the pose is flushed, or the reveal uncovers
    // a T-pose.
    const applied: ClipInfo[] = []
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry, {
      onClipApplied: (info) => applied.push(info),
    })

    await controller.transitionTo('idle')

    expect(applied).toHaveLength(1)
    expect(applied[0].state).toBe('idle')
    expect(applied[0].tracks).toBe(1)
    expect(applied[0].duration).toBe(1)
  })
})

describe('AnimationController — concurrency', () => {
  it('drops a transition that a newer one superseded', async () => {
    // Two motion-ready events arriving back to back. Without the generation
    // check the slower load would land last and overwrite the newer state.
    const { registry, hold, release } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    await controller.transitionTo('idle')

    hold()
    const stale = controller.transitionTo('exercise')
    const fresh = controller.transitionTo('thinking_intro')
    release()

    expect(await stale).toBe(false)
    expect(await fresh).toBe(true)
    expect(controller.currentState).toBe('thinking_intro')
  })

  it('ignores transitions after dispose', async () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    controller.dispose()

    expect(await controller.transitionTo('greeting')).toBe(false)
  })

  it('does not apply a transition that resolves after dispose', async () => {
    const { registry, hold, release } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    hold()
    const pending = controller.transitionTo('greeting')
    controller.dispose()
    release()

    expect(await pending).toBe(false)
    expect(controller.hasPose).toBe(false)
  })
})

describe('AnimationController — one-shot completion', () => {
  it('advances to the successor declared in the FSM', async () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    const seen: CharState[] = []
    controller.on('finished', (state) => seen.push(state))

    await controller.transitionTo('greeting')

    // Run past the clip's 1s duration so the mixer emits 'finished'.
    controller.update(1.5)
    await Promise.resolve()
    await Promise.resolve()

    expect(seen).toEqual(['greeting'])
    expect(controller.currentState).toBe('idle')
  })

  it('freezes on the last frame when the state holds it', async () => {
    // thinking_intro has onFinished: null + holdsLastFrame — it waits for the
    // answer to arrive rather than dropping back to idle on its own.
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)

    await controller.transitionTo('thinking_intro')
    controller.update(1.5)
    await Promise.resolve()

    expect(controller.currentState).toBe('thinking_intro')
  })
})

describe('AnimationController — housekeeping', () => {
  it('retires faded-out actions instead of accumulating them', async () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    await controller.transitionTo('idle')
    await controller.transitionTo('thinking_intro')

    const fading = () => (controller as unknown as { fading: unknown[] }).fading
    expect(fading().length).toBe(1)

    // Past the cross-fade, the outgoing action reaches zero weight.
    for (let i = 0; i < 60; i++) controller.update(1 / 60)

    expect(fading().length).toBe(0)
  })

  it('notifies state listeners and can unsubscribe', async () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    const seen: CharState[] = []
    const off = controller.on('stateChanged', (state) => seen.push(state))

    await controller.transitionTo('idle')
    off()
    await controller.transitionTo('thinking_intro')

    expect(seen).toEqual(['idle'])
  })

  it('stops updating after dispose', async () => {
    const { registry } = makeRegistry()
    const controller = new AnimationController(vrm, registry)
    await controller.transitionTo('idle')
    controller.dispose()

    expect(controller.hasPose).toBe(false)
    expect(() => controller.update(0.016)).not.toThrow()
  })
})
