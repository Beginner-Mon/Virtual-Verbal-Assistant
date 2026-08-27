import type { BodyPart } from './bodyParts'

/**
 * What the user did — the vocabulary that sits between the UI and the avatar.
 *
 * Layer 1 of three. This module knows nothing about three.js, the animation FSM,
 * or the VRM: it describes an INTERACTION, never a reaction. The UI raises an
 * activity; which gesture or emotion that produces is decided per model by the
 * profile (layer 2) and carried out by ActivityDispatcher (layer 3).
 *
 * The separation is the point. It is what lets a character bring its own
 * animation set from the database without the click handler learning any of
 * their names, and what lets a new trigger be added without touching the
 * animation code at all.
 */
export type UserActivity = { kind: 'bodyPartClick'; part: BodyPart }
// Future kinds slot in here — `{ kind: 'chatSent' }`, `{ kind: 'idleElapsed' }`,
// `{ kind: 'ttsStarted' }`. Each needs a case in `activityKey` and nothing else.

export function bodyPartClick(part: BodyPart): UserActivity {
  return { kind: 'bodyPartClick', part }
}

/**
 * Stable lookup key for an activity, e.g. `bodyPartClick:mouth`.
 *
 * This string IS the binding contract: profiles key their `reactions` on it, and
 * those profiles can live in the database. Changing the shape silently unbinds
 * every stored profile rather than failing loudly — hence the test that pins the
 * exact format.
 */
export function activityKey(activity: UserActivity): string {
  switch (activity.kind) {
    case 'bodyPartClick':
      return `bodyPartClick:${activity.part}`
  }
}
