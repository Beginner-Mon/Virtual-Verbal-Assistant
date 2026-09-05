import { describe, expect, it } from 'vitest'
import { FALLBACK_UI_STRINGS, getTimeSlot, uiStringsFor, type UiStrings } from './characterCopy'
import type { Character } from './characters'

/**
 * Vietnamese-exclusive codepoints. Deliberately excludes é è ê à â ô î û ï ç ü
 * ñ ö — every diacritic an English text can legitimately carry through a
 * loanword (café, résumé, naïve, façade). Mirrors the backend's
 * `shared/lang.py:_VN_EXCLUSIVE`, which explains the reasoning at length.
 *
 * Used here as the assertion for "this string is not Vietnamese": counting
 * diacritics generally would flag an English string containing "café".
 */
const VN_EXCLUSIVE =
  /[ăằắẳẵặấầẩẫậđếềểễệốồổỗộơớờởỡợưứừửữựảẻỉỏủỷạẹịọụỵẽĩũỹý]/i

function vietnameseIn(ui: UiStrings): string[] {
  const found: string[] = []
  for (const [key, value] of Object.entries(ui)) {
    if (typeof value === 'string') {
      if (VN_EXCLUSIVE.test(value)) found.push(`${key}: ${value}`)
    } else {
      for (const [slot, text] of Object.entries(value as Record<string, string>)) {
        if (VN_EXCLUSIVE.test(text)) found.push(`${key}.${slot}: ${text}`)
      }
    }
  }
  return found
}

/**
 * A character as the catalog serves it: `ui_strings` keyed by language, because
 * each character is authored once per language in `personas/<slug>/<lang>.md`.
 *
 * Only `vi` here on purpose — a character may ship one language before the
 * other, and the English case below asserts what happens then.
 */
function vietnameseCharacter(): Character {
  return {
    slug: 'anne',
    ui_strings: {
      vi: {
        greeting: {
          morning: 'Chào buổi sáng! Mình là Anne.',
          afternoon: 'Chào buổi chiều! Anne đây.',
          evening: 'Chào buổi tối! Mình là Anne.',
          night: 'Khuya rồi vẫn còn thức à?',
        },
        placeholder: 'Nhắn cho Anne...',
        stage_searching: 'Đang lục thư viện...',
        stage_composing: 'Đang soạn cho bạn...',
        error_stream: 'Mất kết nối giữa chừng rồi.',
      },
    },
  } as unknown as Character
}

describe('uiStringsFor — locale gate', () => {
  it('serves no Vietnamese when the locale is English, even though the character only authored Vietnamese', () => {
    const resolved = uiStringsFor(vietnameseCharacter(), 'en')

    expect(vietnameseIn(resolved)).toEqual([])
  })

  it('keeps the character voice when the locale is Vietnamese', () => {
    const resolved = uiStringsFor(vietnameseCharacter(), 'vi')

    expect(resolved.placeholder).toBe('Nhắn cho Anne...')
    expect(resolved.greeting.morning).toBe('Chào buổi sáng! Mình là Anne.')
    expect(resolved.stage_searching).toBe('Đang lục thư viện...')
  })

  it('falls back per key, not all-or-nothing, when the character authored only some strings', () => {
    const partial = {
      slug: 'x',
      ui_strings: { vi: { placeholder: 'Nhắn cho X...' } },
    } as unknown as Character

    const resolved = uiStringsFor(partial, 'vi')

    expect(resolved.placeholder).toBe('Nhắn cho X...')
    expect(resolved.stage_searching).toBe(FALLBACK_UI_STRINGS.vi.stage_searching)
  })

  it('serves neutral copy for a row still carrying the old un-keyed shape', () => {
    // What a catalog row looks like mid-deploy, before sync_personas_to_db has
    // rewritten it: strings at the top level with no language key. Falling back
    // to neutral copy is the safe direction — the alternative is showing
    // Vietnamese to an English reader, which is the bug this work removed.
    const legacy = {
      slug: 'x',
      ui_strings: { placeholder: 'Nhắn cho X...' },
    } as unknown as Character

    expect(uiStringsFor(legacy, 'en').placeholder).toBe(FALLBACK_UI_STRINGS.en.placeholder)
    expect(uiStringsFor(legacy, 'vi').placeholder).toBe(FALLBACK_UI_STRINGS.vi.placeholder)
  })

  it('falls back to English copy with no character at all', () => {
    const resolved = uiStringsFor(null, 'en')

    expect(vietnameseIn(resolved)).toEqual([])
    expect(resolved.placeholder).toBe(FALLBACK_UI_STRINGS.en.placeholder)
  })

  it('falls back to Vietnamese copy with no character at all', () => {
    const resolved = uiStringsFor(null, 'vi')

    expect(resolved.placeholder).toBe(FALLBACK_UI_STRINGS.vi.placeholder)
  })
})

describe('FALLBACK_UI_STRINGS', () => {
  it('carries the same keys in both locales, so no locale can render undefined', () => {
    expect(Object.keys(FALLBACK_UI_STRINGS.en).sort()).toEqual(
      Object.keys(FALLBACK_UI_STRINGS.vi).sort(),
    )
    expect(Object.keys(FALLBACK_UI_STRINGS.en.greeting).sort()).toEqual(
      Object.keys(FALLBACK_UI_STRINGS.vi.greeting).sort(),
    )
  })

  it('has no Vietnamese in the English bundle', () => {
    expect(vietnameseIn(FALLBACK_UI_STRINGS.en)).toEqual([])
  })
})

describe('getTimeSlot', () => {
  it.each([
    [6, 'morning'],
    [11, 'morning'],
    [12, 'afternoon'],
    [17, 'afternoon'],
    [18, 'evening'],
    [21, 'evening'],
    [22, 'night'],
    [4, 'night'],
  ])('maps hour %i to %s', (hour, slot) => {
    const at = new Date(2026, 0, 1, hour, 0, 0)
    expect(getTimeSlot(at)).toBe(slot)
  })
})
