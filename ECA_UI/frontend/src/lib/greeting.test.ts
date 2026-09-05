import { describe, expect, it } from 'vitest'
import { withGreeting } from './greeting'

const G = '1'

type Msg = { id: string; content: string }

const greeting = (content: string): Msg => ({ id: G, content })
const said = (content: string): Msg => ({ id: 'x', content })

describe('withGreeting', () => {
  it('rewrites the opening line while it is the only thing on screen', () => {
    const out = withGreeting([greeting('Chào buổi sáng!')], G, 'Good morning!')
    expect(out.map((m) => m.content)).toEqual(['Good morning!'])
  })

  it('leaves it alone once a conversation exists', () => {
    // The rule the character-switch path already follows: rewriting a line that
    // has been sitting above real messages makes the screen claim something was
    // said that never was.
    const before = [greeting('Chào buổi sáng!'), said('tôi bị đau lưng')]
    expect(withGreeting(before, G, 'Good morning!')).toBe(before)
  })

  it('returns the SAME array when nothing would change', () => {
    // Identity, not just equality: this feeds setMessages, and a fresh array
    // re-renders the whole transcript on every locale read for no reason.
    const before = [greeting('Good morning!')]
    expect(withGreeting(before, G, 'Good morning!')).toBe(before)
  })

  it('leaves a transcript whose first line is not the greeting alone', () => {
    const before = [said('hello'), said('hi')]
    expect(withGreeting(before, G, 'Good morning!')).toBe(before)
  })

  it('handles an empty transcript without inventing a message', () => {
    const before: Msg[] = []
    expect(withGreeting(before, G, 'Good morning!')).toBe(before)
  })

  it('refuses to blank the greeting', () => {
    // A locale with no copy for this character resolves to '' upstream. Showing
    // an empty first bubble is worse than showing the previous language.
    const before = [greeting('Chào buổi sáng!')]
    expect(withGreeting(before, G, '')).toBe(before)
    expect(withGreeting(before, G, '   ')).toBe(before)
  })
})
