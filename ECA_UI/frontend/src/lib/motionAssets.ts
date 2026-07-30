/**
 * Motion asset index — the ONLY place that globs animation files.
 *
 * Two consumers with different needs, one source:
 *   - AnimationRegistry: resolve a state's `match` RegExp → bundled URL.
 *   - MotionControlPanel: list every file for the debug selector (the offline
 *     verify path for the Kimodo NPZ→BVH pipeline; see plan §4.3).
 *
 * Vite rewrites these globs at build time, so the URLs are content-hashed —
 * matching must be done on the *source path* (`label`), never on the final URL.
 */

const BVH_MODULES = import.meta.glob('../asset/motions/**/*.bvh', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

const FBX_MODULES = import.meta.glob('../asset/motions/**/*.fbx', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

export interface MotionFile {
  /** Source path, e.g. `motions/built-in/action_greeting.bvh`. Match against this. */
  label: string
  /** Bundled (hashed) URL to fetch. */
  url: string
}

export const MOTION_FILES: MotionFile[] = Object.entries({ ...BVH_MODULES, ...FBX_MODULES })
  .map(([assetPath, url]) => ({
    label: assetPath.replace(/^\.\.\/asset\//, '').replace(/\\/g, '/'),
    url,
  }))
  .sort((a, b) => a.label.localeCompare(b.label))

/**
 * First file whose source path matches. Returns null when the asset is missing
 * (renamed/deleted) — callers must handle it rather than render a bind pose.
 */
export function resolveMotionUrl(match: RegExp): string | null {
  return MOTION_FILES.find((f) => match.test(f.label))?.url ?? null
}
