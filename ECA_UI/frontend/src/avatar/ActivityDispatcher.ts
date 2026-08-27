import type { AvatarController } from './AvatarController'
import type { GestureDef, Reaction } from './AvatarProfile'
import type { AnimationController } from '../lib/AnimationController'
import type { AnimationRegistry, DynamicClip } from '../lib/AnimationRegistry'
import { activityKey, type UserActivity } from './userActivity'

/**
 * Turns what the user did into what this character does about it.
 *
 * Layer 3 of three, and the only place that knows both halves. The UI raises a
 * `UserActivity` and knows no animation names; the profile declares reactions
 * and knows nothing about the FSM or the mixer; this class joins them.
 *
 * Emotion and animation deliberately run through here TOGETHER. They are two
 * fields of one `Reaction`, resolved from one per-character profile, so a
 * character gains a new reaction — expression, movement, or both — by shipping
 * data, not by anyone editing a click handler.
 *
 * Everything is read through getters rather than captured in the constructor:
 * the avatar controller attaches after a network round-trip, the animation
 * controller is rebuilt per VRM, and holding a stale reference to either is how
 * a click ends up driving a model that is no longer on screen.
 */
export interface ActivityDispatcherDeps {
  getAvatar: () => AvatarController | null
  getAnim: () => AnimationController | null
  getRegistry: () => AnimationRegistry | null
  /** Bundled motion path -> hashed URL. `resolveMotionUrl` in motionAssets. */
  resolveBuiltIn: (match: string) => string | null
}

export class ActivityDispatcher {
  private readonly deps: ActivityDispatcherDeps

  constructor(deps: ActivityDispatcherDeps) {
    this.deps = deps
  }

  /**
   * Run whatever this character binds to `activity`. Resolves true when
   * something actually happened, so callers can log or measure it.
   */
  async dispatch(activity: UserActivity): Promise<boolean> {
    const avatar = this.deps.getAvatar()
    if (!avatar) return false

    const reaction: Reaction | undefined = avatar.profile.reactions?.[activityKey(activity)]
    if (!reaction) return false

    let acted = false

    if (reaction.emotion) {
      const { name, intensity = 1, durationMs } = reaction.emotion
      avatar.setEmotion(name, intensity, durationMs)
      acted = true
    }

    if (reaction.gesture) {
      acted = (await this.playGesture(reaction.gesture)) || acted
    }

    return acted
  }

  /**
   * Warm this character's gestures during idle time, so the first click does not
   * pay the fetch-and-retarget stall (111-214 ms, measured in AnimationRegistry).
   *
   * Only possible because gestures are declared per character rather than named
   * at the moment of the click.
   */
  prefetch(): void {
    const avatar = this.deps.getAvatar()
    const registry = this.deps.getRegistry()
    if (!avatar || !registry) return

    const clips: DynamicClip[] = []
    for (const gesture of Object.values(avatar.profile.gestures ?? {})) {
      const clip = this.resolveGesture(gesture)
      if (clip) clips.push(clip)
    }
    if (clips.length > 0) registry.prefetchGestures(clips)
  }

  private async playGesture(id: string): Promise<boolean> {
    const avatar = this.deps.getAvatar()
    const anim = this.deps.getAnim()
    const registry = this.deps.getRegistry()
    if (!avatar || !anim || !registry) return false

    const gesture = avatar.profile.gestures?.[id]
    if (!gesture) {
      // A reaction naming a gesture the character does not have. Common while a
      // database profile and the bundled defaults drift apart, so it warns
      // rather than throws — the emotion half of the reaction still ran.
      console.warn(`[activity] "${avatar.profile.modelId}" has no gesture "${id}"`)
      return false
    }

    const clip = this.resolveGesture(gesture)
    if (!clip) {
      console.warn(`[activity] gesture "${id}" has no resolvable clip`)
      return false
    }

    // Registry first. `transitionTo` resolves the clip THROUGH the registry, so
    // registering afterwards would play whichever gesture ran last.
    registry.update('gesture', clip)
    return anim.transitionTo('gesture')
  }

  private resolveGesture(gesture: GestureDef): DynamicClip | null {
    if ('url' in gesture.source) {
      return { url: gesture.source.url, loader: gesture.source.loader }
    }
    const url = this.deps.resolveBuiltIn(gesture.source.builtIn)
    if (!url) return null
    // Bundled motions are Mixamo FBX; BVH ones come from the generated pipeline
    // and arrive as absolute URLs instead.
    return { url, loader: 'fbx' }
  }
}
