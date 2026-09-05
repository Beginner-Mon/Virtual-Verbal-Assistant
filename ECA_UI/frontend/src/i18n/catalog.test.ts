import { describe, expect, it } from 'vitest'
import en from './locales/en.json'
import vi from './locales/vi.json'
import { LOCALES } from './locale'

/**
 * Same codepoint set the backend uses to decide "this is Vietnamese"
 * (`shared/lang.py:_VN_EXCLUSIVE`): the diacritics that exist in no other Latin
 * orthography. Excludes é è ê à â ô ç ü ñ, which an English string can carry
 * legitimately through café / naïve / façade.
 */
const VN_EXCLUSIVE =
  /[ăằắẳẵặấầẩẫậđếềểễệốồổỗộơớờởỡợưứừửữựảẻỉỏủỷạẹịọụỵẽĩũỹý]/i

type Tree = { [key: string]: string | Tree }

function flatten(tree: Tree, prefix = ''): Map<string, string> {
  const out = new Map<string, string>()
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (typeof value === 'string') out.set(path, value)
    else for (const [k, v] of flatten(value, path)) out.set(k, v)
  }
  return out
}

const flatEn = flatten(en as Tree)
const flatVi = flatten(vi as Tree)

describe('locale catalogues', () => {
  it('covers every supported locale', () => {
    expect(LOCALES).toEqual(['en', 'vi'])
  })

  it('has no key in English that Vietnamese is missing', () => {
    const missing = [...flatEn.keys()].filter((k) => !flatVi.has(k))
    expect(missing).toEqual([])
  })

  it('has no key in Vietnamese that English is missing', () => {
    const orphaned = [...flatVi.keys()].filter((k) => !flatEn.has(k))
    expect(orphaned).toEqual([])
  })

  it('has no empty values', () => {
    const empty = [
      ...[...flatEn].filter(([, v]) => !v.trim()).map(([k]) => `en:${k}`),
      ...[...flatVi].filter(([, v]) => !v.trim()).map(([k]) => `vi:${k}`),
    ]
    expect(empty).toEqual([])
  })

  it('has no Vietnamese text in the English catalogue', () => {
    const leaked = [...flatEn].filter(([, v]) => VN_EXCLUSIVE.test(v)).map(([k, v]) => `${k}: ${v}`)
    expect(leaked).toEqual([])
  })

  it('has no untranslated Vietnamese entries left identical to English', () => {
    // A value copied verbatim from en.json into vi.json is the shape a
    // half-finished translation takes: the key exists, so the parity checks
    // above pass, and the screen still shows English.
    //
    // Some values are legitimately identical. They are listed rather than
    // pattern-matched, because "this reads the same in both languages" is a
    // judgement someone should make once and leave a record of — a heuristic
    // like "skip anything without a space" silently forgives real omissions.
    const IDENTICAL_ON_PURPOSE = new Set([
      'about.product', // product name
      'about.studio', // company name
      'notifications.version', // product name + version
      'chat.chip_web', // "Web" is the same word in both
    ])

    const untranslated = [...flatVi]
      .filter(([k, v]) => !IDENTICAL_ON_PURPOSE.has(k) && flatEn.get(k) === v)
      .map(([k, v]) => `${k}: ${v}`)
    expect(untranslated).toEqual([])
  })
})
