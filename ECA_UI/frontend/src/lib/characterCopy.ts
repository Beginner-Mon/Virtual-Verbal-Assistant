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

export interface UiStrings {
  greeting: string
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
}

/**
 * Used when the catalog has not loaded yet, or has not been redeployed with the
 * `ui_strings` column. Deliberately character-neutral: a wrong name is worse
 * than no name. Vietnamese, because that is the product's primary language, and
 * no emoji anywhere.
 */
export const FALLBACK_UI_STRINGS: UiStrings = {
  greeting: 'Xin chào! Hôm nay mình giúp được gì cho bạn?',
  placeholder: 'Nhập tin nhắn...',
  stage_searching: 'Đang tìm kiếm thông tin...',
  stage_composing: 'Đang soạn câu trả lời...',
  error_stream: 'Đã có lỗi xảy ra. Bạn thử gửi lại nhé.',
  motion_rendering: 'Đang dựng động tác...',
  motion_unavailable: 'Tính năng dựng động tác đang tạm tắt.',
  motion_busy: 'Đang có nhiều yêu cầu dựng động tác, bạn thử lại sau nhé.',
  motion_failed: 'Mình chưa dựng được động tác này.',
}

/**
 * Resolve one character's copy, filling any missing key from the fallback.
 *
 * Per-key rather than all-or-nothing: a persona that defines a greeting but no
 * stage labels should keep its greeting, not lose it because the object was
 * incomplete.
 */
export function uiStringsFor(character?: Character | null): UiStrings {
  const authored = character?.ui_strings ?? {}
  const resolved = { ...FALLBACK_UI_STRINGS }
  for (const key of Object.keys(FALLBACK_UI_STRINGS) as (keyof UiStrings)[]) {
    const value = authored[key]
    if (typeof value === 'string' && value.trim()) resolved[key] = value
  }
  return resolved
}
