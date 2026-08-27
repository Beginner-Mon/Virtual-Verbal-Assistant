import { describe, expect, it, vi } from 'vitest'
import type { AvatarProfile } from './AvatarProfile'
import { defaultProfile } from './profiles/default'
import { ActivityDispatcher } from './ActivityDispatcher'
import { bodyPartClick } from './userActivity'

/**
 * The dispatcher is the only place that knows both sides — emotion and
 * animation — so these tests are mostly about what it does NOT do: it must not
 * fire an animation for a binding that only asks for an emotion, and it must not
 * ask the FSM for a gesture the character does not have.
 */

function makeDeps(profile: AvatarProfile) {
  const avatar = { profile, setEmotion: vi.fn() }
  const registry = { update: vi.fn(), prefetchGestures: vi.fn() }
  const anim = { transitionTo: vi.fn(async () => true) }
  const resolveBuiltIn = vi.fn((match: string) => `/assets/${match}-hash.fbx`)

  const dispatcher = new ActivityDispatcher({
    getAvatar: () => avatar as never,
    getAnim: () => anim as never,
    getRegistry: () => registry as never,
    resolveBuiltIn,
  })

  return { dispatcher, avatar, registry, anim, resolveBuiltIn }
}

const withProfile = (extra: Partial<AvatarProfile>): AvatarProfile => ({
  ...defaultProfile,
  ...extra,
})

describe('ActivityDispatcher', () => {
  it('plays the gesture and lifts the expression from one binding', async () => {
    const { dispatcher, avatar, registry, anim } = makeDeps(defaultProfile)

    const acted = await dispatcher.dispatch(bodyPartClick('mouth'))

    expect(acted).toBe(true)
    expect(avatar.setEmotion).toHaveBeenCalledWith('happy', 1, 600)
    // Registry before transition: transitionTo resolves the clip through it, so
    // registering afterwards would play whatever ran last.
    expect(registry.update).toHaveBeenCalledBefore(anim.transitionTo as never)
    expect(anim.transitionTo).toHaveBeenCalledWith('gesture')
  })

  it('does nothing for an activity the character has no binding for', async () => {
    const { dispatcher, avatar, anim } = makeDeps(defaultProfile)

    const acted = await dispatcher.dispatch(bodyPartClick('leftFoot'))

    expect(acted).toBe(false)
    expect(avatar.setEmotion).not.toHaveBeenCalled()
    expect(anim.transitionTo).not.toHaveBeenCalled()
  })

  it('runs an emotion-only binding without touching the FSM', async () => {
    const profile = withProfile({
      reactions: { 'bodyPartClick:head': { emotion: { name: 'surprised' } } },
    })
    const { dispatcher, avatar, anim } = makeDeps(profile)

    expect(await dispatcher.dispatch(bodyPartClick('head'))).toBe(true)
    expect(avatar.setEmotion).toHaveBeenCalledWith('surprised', 1, undefined)
    expect(anim.transitionTo).not.toHaveBeenCalled()
  })

  it('runs a gesture-only binding without touching the expression', async () => {
    const profile = withProfile({
      reactions: { 'bodyPartClick:head': { gesture: 'kiss' } },
    })
    const { dispatcher, avatar, anim } = makeDeps(profile)

    expect(await dispatcher.dispatch(bodyPartClick('head'))).toBe(true)
    expect(avatar.setEmotion).not.toHaveBeenCalled()
    expect(anim.transitionTo).toHaveBeenCalledWith('gesture')
  })

  it('refuses a reaction pointing at a gesture the character does not declare', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const profile = withProfile({
      gestures: {},
      reactions: { 'bodyPartClick:mouth': { gesture: 'kiss', emotion: { name: 'happy' } } },
    })
    const { dispatcher, avatar, anim } = makeDeps(profile)

    await dispatcher.dispatch(bodyPartClick('mouth'))

    // The emotion still runs — half a reaction beats none — but the FSM is never
    // asked for a clip that cannot be resolved.
    expect(avatar.setEmotion).toHaveBeenCalled()
    expect(anim.transitionTo).not.toHaveBeenCalled()
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })

  it('refuses a gesture whose bundled asset is missing', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { dispatcher, anim, resolveBuiltIn } = makeDeps(defaultProfile)
    resolveBuiltIn.mockReturnValue(null as never)

    await dispatcher.dispatch(bodyPartClick('mouth'))

    expect(anim.transitionTo).not.toHaveBeenCalled()
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })

  it('does nothing at all before the avatar has attached', async () => {
    const dispatcher = new ActivityDispatcher({
      getAvatar: () => null,
      getAnim: () => null,
      getRegistry: () => null,
      resolveBuiltIn: () => null,
    })

    expect(await dispatcher.dispatch(bodyPartClick('mouth'))).toBe(false)
  })

  it('warms only the gestures the character actually declares', () => {
    const { dispatcher, registry } = makeDeps(defaultProfile)

    dispatcher.prefetch()

    const warmed = registry.prefetchGestures.mock.calls[0][0] as { url: string }[]
    expect(warmed).toHaveLength(Object.keys(defaultProfile.gestures ?? {}).length)
    expect(warmed[0].url).toContain('kiss')
  })
})
