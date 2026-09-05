/**
 * Chat-surface copy for the selected character.
 *
 * Everything the user reads that does NOT come out of the model used to be a
 * hard-coded English string: the opening greeting introduced "ECA" whichever
 * avatar was on screen, the stage labels and the error line never moved. Those
 * strings now travel with the character.
 *
 * Authored in `agenticRAG/langgraph_agents/personas/<slug>.md` under
 * `## UI Strings`, pushed to `characters.ui_strings` by
 * scripts/sync_personas_to_db.py, and served by the catalog Lambda. One file
 * per character defines both how they speak and how the UI speaks for them.
 *
 * `ui_strings` is a separate column from `persona` on purpose — persona is the
 * system prompt and is never served to a browser.
 */

import type { Character } from './characters'
import { DEFAULT_LOCALE, type Locale } from '@/i18n/locale'

export type GreetingSlots = {
  morning: string
  afternoon: string
  evening: string
  night: string
}

export type TimeSlot = keyof GreetingSlots

export interface UiStrings {
  greeting: GreetingSlots
  placeholder: string
  stage_searching: string
  stage_composing: string
  error_stream: string
  /* Motion renders. The GPU worker is scaled to zero by default and switched on
   * deliberately, so `motion_unavailable` is the ordinary path rather than an
   * error — a user who asked to SEE a movement and gets only text otherwise has
   * no way to tell whether the request was understood. */
  motion_rendering: string
  motion_unavailable: string
  motion_busy: string
  motion_failed: string
  /** Restored history: the render is past its storage deadline and cannot be
   *  fetched again. Distinct from motion_failed — it did not fail, it aged. */
  motion_gone: string
}

/**
 * Used when the catalog has not loaded yet, has not been redeployed with the
 * `ui_strings` column, or — the case this became per-locale for — when the
 * viewer reads the site in a language the character did not author.
 *
 * Deliberately character-neutral: a wrong name is worse than no name.
 * No emoji anywhere.
 */
export const FALLBACK_UI_STRINGS: Record<Locale, UiStrings> = {
  en: {
    greeting: {
      morning: 'Good morning! What can I help you with today?',
      afternoon: 'Good afternoon! How are you doing?',
      evening: 'Good evening! I am here with you.',
      night: 'Still up at this hour? I am here if you need me.',
    },
    placeholder: 'Type a message...',
    stage_searching: 'Searching...',
    stage_composing: 'Writing a reply...',
    error_stream: 'Something went wrong. Please send that again.',
    motion_rendering: 'Building the movement...',
    motion_unavailable: 'Movement rendering is switched off right now.',
    motion_busy: 'Too many movement requests at the moment — please try again shortly.',
    motion_failed: 'I could not build this movement.',
    motion_gone: 'The movement for this message is past its storage deadline.',
  },
  vi: {
    greeting: {
      morning: 'Chào buổi sáng! Hôm nay mình giúp gì cho bạn?',
      afternoon: 'Chào buổi chiều! Bạn đang thế nào rồi?',
      evening: 'Chào buổi tối! Mình ở đây với bạn.',
      night: 'Khuya rồi, bạn vẫn chưa ngủ à? Mình ở đây nhé.',
    },
    placeholder: 'Nhập tin nhắn...',
    stage_searching: 'Đang tìm kiếm thông tin...',
    stage_composing: 'Đang soạn câu trả lời...',
    error_stream: 'Đã có lỗi xảy ra. Bạn thử gửi lại nhé.',
    motion_rendering: 'Đang dựng động tác...',
    motion_unavailable: 'Tính năng dựng động tác đang tạm tắt.',
    motion_busy: 'Đang có nhiều yêu cầu dựng động tác, bạn thử lại sau nhé.',
    motion_failed: 'Mình chưa dựng được động tác này.',
    motion_gone: 'Động tác của câu này đã hết hạn lưu trữ.',
  },
}

/**
 * 05-11: morning, 12-17: afternoon, 18-21: evening, 22-04: night
 *
 * The `h >= 5` guard on the night branch is load-bearing, and its absence was a
 * bug: with only `h < 18` deciding afternoon, every hour from midnight to 04:59
 * fell through to "afternoon" and greeted a 3am visitor with "Chào buổi chiều".
 * The docstring above always said 22-04 was night; the code only implemented
 * the 22-23 half of that range.
 */
export function getTimeSlot(date = new Date()): TimeSlot {
  const h = date.getHours()
  if (h < 5) return 'night'
  if (h < 12) return 'morning'
  if (h < 18) return 'afternoon'
  if (h < 22) return 'evening'
  return 'night'
}

export function getGreeting(ui: UiStrings, now = new Date()): string {
  const slot = getTimeSlot(now)
  return ui.greeting[slot] ?? FALLBACK_UI_STRINGS[DEFAULT_LOCALE].greeting[slot]
}

/**
 * Resolve one character's copy for the locale the site is being read in.
 *
 * The catalog now serves `ui_strings` keyed by language — `{vi: {...}, en: {...}}`
 * — because each character is authored once per language
 * (`personas/<slug>/<lang>.md`). Pick the bundle for this locale; a character
 * with no bundle for it falls back to the neutral copy, which is a real state
 * rather than a placeholder: a character may ship one language before the other.
 *
 * This replaced a deliberate Phase 1 degradation. While the backend served only
 * Vietnamese, an English reader was given neutral copy instead of the
 * character's voice, on the grounds that a Vietnamese greeting inside an English
 * UI is worse than a characterless one. That trade is over — the voice is back.
 *
 * Per-key rather than all-or-nothing: a persona that defines a greeting but no
 * stage labels should keep its greeting, not lose it because the object was
 * incomplete. Greeting is an object with 4 slots — each slot falls back
 * individually, so a persona that only defined morning still gets a usable
 * afternoon/evening/night.
 */
export function uiStringsFor(
  character?: Character | null,
  locale: Locale = DEFAULT_LOCALE,
): UiStrings {
  const fallback = FALLBACK_UI_STRINGS[locale] ?? FALLBACK_UI_STRINGS[DEFAULT_LOCALE]
  const resolved: UiStrings = {
    ...fallback,
    greeting: { ...fallback.greeting },
  }

  // `{vi: {...}, en: {...}}` from the catalog. A row still carrying the old flat
  // shape has no key for this locale, so it resolves to neutral copy rather than
  // to the wrong language — which is the safe direction to fail during a deploy.
  const byLocale = (character?.ui_strings ?? {}) as Record<string, unknown>
  const bundle = byLocale[locale]
  const authored = (
    bundle && typeof bundle === 'object' && !Array.isArray(bundle) ? bundle : {}
  ) as Record<string, unknown>

  for (const key of Object.keys(fallback) as (keyof UiStrings)[]) {
    if (key === 'greeting') {
      const g = authored['greeting']
      if (g && typeof g === 'object' && !Array.isArray(g)) {
        const obj = g as Record<string, unknown>
        for (const slot of Object.keys(fallback.greeting) as TimeSlot[]) {
          const v = obj[slot]
          if (typeof v === 'string' && v.trim()) {
            resolved.greeting[slot] = v
          }
        }
      }
      // No string fallback — greeting is now object-only per spec (bỏ fallback)
      continue
    }
    const value = authored[key]
    if (typeof value === 'string' && value.trim()) {
      ;(resolved as unknown as Record<string, unknown>)[key] = value
    }
  }
  return resolved
}
