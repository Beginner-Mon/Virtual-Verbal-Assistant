/**
 * The rules that stop the app asking the server about a conversation that does
 * not exist.
 *
 * The first test is the one that matters: reading an empty store must return
 * null AND write nothing. The version this replaced minted a UUID at that
 * moment and stored it, so the very next thing the app did was fetch an id it
 * had invented a millisecond earlier — a 404 on every clean load, and a
 * permanent one for anyone who pressed "new chat" without sending a message.
 */

import { describe, expect, it } from 'vitest'
import {
  SESSION_KEY,
  SESSION_TTL_MS,
  clearSessionPointer,
  readSessionPointer,
  stampSessionPointer,
} from './chatSession'

/** A Storage that records writes, so "wrote nothing" is assertable. */
function fakeStorage(initial: Record<string, string> = {}): Storage & { writes: number } {
  const map = new Map(Object.entries(initial))
  return {
    writes: 0,
    get length() {
      return map.size
    },
    key: (i: number) => Array.from(map.keys())[i] ?? null,
    getItem: (k: string) => map.get(k) ?? null,
    setItem(this: { writes: number }, k: string, v: string) {
      this.writes++
      map.set(k, v)
    },
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
  } as Storage & { writes: number }
}

function throwingStorage(): Storage {
  const boom = () => {
    throw new DOMException('denied', 'SecurityError')
  }
  return {
    length: 0,
    key: boom,
    getItem: boom,
    setItem: boom,
    removeItem: boom,
    clear: boom,
  } as unknown as Storage
}

const NOW = 1_757_000_000_000
const MINUTE = 60 * 1000

describe('readSessionPointer', () => {
  it('returns null and writes nothing when there is no pointer', () => {
    // The regression this file exists for. A write here means the app has
    // invented a conversation id that nothing asked for, and the next thing it
    // does is fetch that id and get a 404.
    const s = fakeStorage()
    expect(readSessionPointer(NOW, s)).toBeNull()
    expect(s.writes).toBe(0)
    expect(s.getItem(SESSION_KEY)).toBeNull()
  })

  it('returns the id while it is still fresh', () => {
    const s = fakeStorage()
    stampSessionPointer('abc', NOW - MINUTE, s)
    expect(readSessionPointer(NOW, s)).toBe('abc')
  })

  it('returns null and drops the key once the TTL has passed', () => {
    const s = fakeStorage()
    stampSessionPointer('abc', NOW - SESSION_TTL_MS - MINUTE, s)
    expect(readSessionPointer(NOW, s)).toBeNull()
    expect(s.getItem(SESSION_KEY)).toBeNull()
  })

  it('keeps a pointer that is exactly at the boundary', () => {
    const s = fakeStorage()
    stampSessionPointer('abc', NOW - SESSION_TTL_MS, s)
    expect(readSessionPointer(NOW, s)).toBe('abc')
  })

  it('discards the legacy bare-string value', () => {
    // Written by the version before this one. It carries no timestamp, so its
    // age is unknowable — and it may be one of the pointers that never had a
    // row behind it, which is exactly what was found in a real browser.
    const s = fakeStorage({ [SESSION_KEY]: '19c6ccde-25ed-401c-ac34-9f1f15c62f26' })
    expect(readSessionPointer(NOW, s)).toBeNull()
    expect(s.getItem(SESSION_KEY)).toBeNull()
  })

  it('discards malformed JSON without throwing', () => {
    const s = fakeStorage({ [SESSION_KEY]: '{"id": "abc"' })
    expect(readSessionPointer(NOW, s)).toBeNull()
    expect(s.getItem(SESSION_KEY)).toBeNull()
  })

  it.each([
    ['missing at', '{"id":"abc"}'],
    ['missing id', '{"at":1757000000000}'],
    ['empty id', '{"id":"","at":1757000000000}'],
    ['at is not a number', '{"id":"abc","at":"yesterday"}'],
    ['at is NaN', '{"id":"abc","at":null}'],
    ['an array', '[]'],
  ])('discards a pointer with %s', (_label, raw) => {
    const s = fakeStorage({ [SESSION_KEY]: raw })
    expect(readSessionPointer(NOW, s)).toBeNull()
  })

  it('returns null rather than throwing when storage is unavailable', () => {
    // Private mode. An unreadable pointer is the same as no pointer.
    expect(() => readSessionPointer(NOW, throwingStorage())).not.toThrow()
    expect(readSessionPointer(NOW, throwingStorage())).toBeNull()
  })
})

describe('stampSessionPointer', () => {
  it('round-trips through read', () => {
    const s = fakeStorage()
    stampSessionPointer('session-1', NOW, s)
    expect(readSessionPointer(NOW, s)).toBe('session-1')
  })

  it('restarts the clock, so an active conversation never expires mid-use', () => {
    const s = fakeStorage()
    stampSessionPointer('abc', NOW, s)
    // 90 minutes later the user sends another message
    stampSessionPointer('abc', NOW + 90 * MINUTE, s)
    // 90 minutes after THAT: 3 hours from the start, but only 90 minutes from
    // the last message, so it is still the conversation they are in.
    expect(readSessionPointer(NOW + 180 * MINUTE, s)).toBe('abc')
  })

  it('does not throw when storage is unavailable', () => {
    expect(() => stampSessionPointer('abc', NOW, throwingStorage())).not.toThrow()
  })
})

describe('clearSessionPointer', () => {
  it('removes the pointer', () => {
    const s = fakeStorage()
    stampSessionPointer('abc', NOW, s)
    clearSessionPointer(s)
    expect(readSessionPointer(NOW, s)).toBeNull()
  })

  it('is a no-op on an empty store', () => {
    const s = fakeStorage()
    expect(() => clearSessionPointer(s)).not.toThrow()
  })

  it('does not throw when storage is unavailable', () => {
    expect(() => clearSessionPointer(throwingStorage())).not.toThrow()
  })
})
