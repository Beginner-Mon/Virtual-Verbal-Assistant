/**
 * Export the avatar profile registry as JSON on stdout.
 *
 * Consumed by scripts/upload_characters_to_s3.py, which seeds
 * characters.avatar_profile. The profiles are resolved by importing the real
 * modules rather than parsing the files: bronya.ts spreads defaultProfile, so
 * anything that reads a single file in isolation gets a profile with no
 * recipes and no visemes, and the model renders expressionless.
 *
 * Requires Node >= 22.18 (TypeScript type stripping on by default). The type
 * annotations here are erased at load; no build step, no extra dependency.
 *
 * Usage:
 *   node scripts/export-avatar-profiles.mjs            # every profile
 *   node scripts/export-avatar-profiles.mjs bronya     # one, resolved via loadProfile
 */

import { registerHooks } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// The source imports siblings without a file extension ("./profiles/default"),
// which Vite resolves and Node does not. Rather than edit application code to
// suit a build script, teach this process to retry with `.ts` appended.
registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context)
    } catch (err) {
      if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier)) {
        return nextResolve(`${specifier}.ts`, context)
      }
      throw err
    }
  },
})

const scriptDir = dirname(fileURLToPath(import.meta.url))
const avatarDir = resolve(scriptDir, '..', 'src', 'avatar')

const mod = await import(`file://${resolve(avatarDir, 'AvatarProfile.ts')}`)
const { loadProfile } = mod

const requested = process.argv.slice(2)

if (requested.length > 0) {
  // loadProfile falls back to `default` with modelId rewritten, which is
  // exactly what a character with no hand-written override should store.
  const out = {}
  for (const id of requested) out[id] = loadProfile(id)
  process.stdout.write(JSON.stringify(out, null, 2))
} else {
  const { defaultProfile } = await import(`file://${resolve(avatarDir, 'profiles', 'default.ts')}`)
  const { bronyaProfile } = await import(`file://${resolve(avatarDir, 'profiles', 'bronya.ts')}`)
  const { seeleProfile } = await import(`file://${resolve(avatarDir, 'profiles', 'seele.ts')}`)
  process.stdout.write(JSON.stringify({
    default: defaultProfile,
    bronya: bronyaProfile,
    seele: seeleProfile,
  }, null, 2))
}
