import { describe, expect, it } from 'vitest'
import { CHAR_STATES, STATES, canTransition, staticSourceOf } from './AnimationStates'

/**
 * Pins the reachability decisions for `gesture` — the state that carries every
 * per-character animation.
 *
 * Worth a test rather than trusting the config alone: `reach` is one word in one
 * entry, and getting it wrong does not fail the build. It fails by letting a
 * click on the avatar cut off the thinking pose while the user waits for an
 * answer, which is the kind of thing nobody notices until a demo.
 */
describe('gesture reachability', () => {
  it('runs from idle', () => {
    expect(canTransition('idle', 'gesture')).toBe(true)
  })

  it('never interrupts a state the user is waiting on', () => {
    // thinking_intro holds its pose until the LLM answers; exercise is a
    // generated clinical motion; greeting is the boot sequence.
    expect(canTransition('thinking_intro', 'gesture')).toBe(false)
    expect(canTransition('exercise', 'gesture')).toBe(false)
    expect(canTransition('greeting', 'gesture')).toBe(false)
  })

  it('does not interrupt the idle filler either', () => {
    // Decided deliberately: a click that does nothing while the character is
    // clearly mid-animation reads as "busy", not as broken.
    expect(canTransition('bored', 'gesture')).toBe(false)
  })

  it('cannot be spammed into restarting itself', () => {
    expect(canTransition('gesture', 'gesture')).toBe(false)
  })

  it('always has a way home', () => {
    expect(canTransition('gesture', 'idle')).toBe(true)
  })

  it('carries its clip at runtime rather than from a bundled file', () => {
    // A per-character animation set cannot be a compile-time union — this is
    // what keeps CharState closed while the animations stay open.
    expect(staticSourceOf('gesture')).toBeNull()
  })
})

describe('FSM invariants', () => {
  it('every one-shot either continues somewhere or holds deliberately', () => {
    for (const state of CHAR_STATES) {
      const def = STATES[state]
      if (def.loop !== 'once') continue
      const parks = def.onFinished === null && 'holdsLastFrame' in def
      expect(def.onFinished !== null || parks).toBe(true)
    }
  })

  it('every one-shot successor is a real state', () => {
    for (const state of CHAR_STATES) {
      const next = STATES[state].onFinished
      if (next === null) continue
      expect(CHAR_STATES).toContain(next)
    }
  })
})
