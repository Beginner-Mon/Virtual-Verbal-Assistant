import { describe, expect, it } from 'vitest'
import { mergeProfile } from './AvatarProfile'
import { defaultProfile } from './profiles/default'

/**
 * Covers the rule that protects the bundled configuration: a profile fetched
 * from the database OVERRIDES fields, it does not replace the whole record.
 *
 * `loadProfileAsync` used to do `{ ...raw, modelId }`. That was harmless while a
 * remote profile carried every field, but the moment the bundled profile owns
 * something the database row does not — gestures, morphRepairMap, binaryEmotions
 * — a straight spread silently deletes it. A character whose row predates the
 * gesture work would lose its animations rather than inherit the defaults.
 */

const REMOTE_MINIMUM = {
  version: 1 as const,
  modelId: 'ignored',
  recipes: defaultProfile.recipes,
  visemes: defaultProfile.visemes,
  blinkChannel: 'blink',
}

describe('mergeProfile', () => {
  it('keeps bundled gestures when the remote profile has none', () => {
    const merged = mergeProfile(defaultProfile, REMOTE_MINIMUM, 'anne')
    expect(merged.gestures).toEqual(defaultProfile.gestures)
    expect(merged.reactions).toEqual(defaultProfile.reactions)
  })

  it('lets the remote profile bring its own gestures', () => {
    const gestures = { wave: { source: { builtIn: 'wave' } } }
    const reactions = { 'bodyPartClick:leftHand': { gesture: 'wave' } }
    const merged = mergeProfile(defaultProfile, { ...REMOTE_MINIMUM, gestures, reactions }, 'anne')

    expect(merged.gestures).toEqual(gestures)
    expect(merged.reactions).toEqual(reactions)
  })

  it('keeps every other bundled-only field', () => {
    const bundled = {
      ...defaultProfile,
      morphRepairMap: { blink: 'Blink' },
      binaryEmotions: ['relaxed' as const],
      greetingEmotion: 'surprised' as const,
    }
    const merged = mergeProfile(bundled, REMOTE_MINIMUM, 'anne')

    expect(merged.morphRepairMap).toEqual({ blink: 'Blink' })
    expect(merged.binaryEmotions).toEqual(['relaxed'])
    expect(merged.greetingEmotion).toBe('surprised')
  })

  it('prefers remote values where both sides have one', () => {
    const merged = mergeProfile(defaultProfile, { ...REMOTE_MINIMUM, blinkChannel: 'Blink_A' }, 'anne')
    expect(merged.blinkChannel).toBe('Blink_A')
  })

  it('always stamps the requested model id', () => {
    // The remote row carries its own slug; the caller's id is the one the rest of
    // the avatar stack keys on.
    const merged = mergeProfile(defaultProfile, REMOTE_MINIMUM, 'hatsune-miku')
    expect(merged.modelId).toBe('hatsune-miku')
  })

  it('does not mutate either input', () => {
    const bundled = { ...defaultProfile }
    const remote = { ...REMOTE_MINIMUM }
    mergeProfile(bundled, remote, 'anne')

    expect(bundled).toEqual(defaultProfile)
    expect(remote).toEqual(REMOTE_MINIMUM)
  })
})

describe('the bundled default profile', () => {
  it('binds a click on the mouth to the kiss gesture', () => {
    const reaction = defaultProfile.reactions?.['bodyPartClick:mouth']
    expect(reaction?.gesture).toBe('kiss')
    expect(reaction?.emotion?.name).toBe('happy')
  })

  it('declares every gesture its reactions reference', () => {
    // A reaction pointing at a missing gesture id is a dead binding — the click
    // would set an emotion and play nothing.
    for (const reaction of Object.values(defaultProfile.reactions ?? {})) {
      if (!reaction.gesture) continue
      expect(defaultProfile.gestures?.[reaction.gesture]).toBeDefined()
    }
  })
})
