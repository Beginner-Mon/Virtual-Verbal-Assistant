import { describe, expect, it } from 'vitest'
import { BODY_PARTS } from './bodyParts'
import { activityKey, bodyPartClick } from './userActivity'

describe('activityKey', () => {
  it('gives every body part its own key', () => {
    // Reactions are looked up by this string, so a collision would silently make
    // two parts share one animation.
    const keys = BODY_PARTS.map((part) => activityKey(bodyPartClick(part)))
    expect(new Set(keys).size).toBe(BODY_PARTS.length)
  })

  it('is stable — the key is the binding contract, not an implementation detail', () => {
    // Profiles (including ones stored in the database) key their reactions on
    // this string. Changing the shape silently unbinds every existing profile.
    expect(activityKey(bodyPartClick('mouth'))).toBe('bodyPartClick:mouth')
    expect(activityKey(bodyPartClick('leftEye'))).toBe('bodyPartClick:leftEye')
  })

  it('round-trips the same activity to the same key', () => {
    expect(activityKey(bodyPartClick('head'))).toBe(activityKey(bodyPartClick('head')))
  })
})
