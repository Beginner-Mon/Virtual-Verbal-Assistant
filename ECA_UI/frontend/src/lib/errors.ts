/**
 * Extract a human-readable message from an unknown throw.
 * Replaces `catch (err: any) { err.message }` which hides the `unknown` that
 * `catch` actually gives you. Use with `catch (e: unknown)`.
 */
export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message
  if (typeof e === 'string') return e
  // Amplify / AWS errors often carry `message` as a plain prop, not an Error
  if (e && typeof e === 'object' && 'message' in e && typeof (e as { message: unknown }).message === 'string') {
    return (e as { message: string }).message
  }
  try {
    return JSON.stringify(e)
  } catch {
    return String(e)
  }
}

export function errorName(e: unknown): string {
  if (e instanceof Error) return e.name
  if (e && typeof e === 'object' && 'name' in e && typeof (e as { name: unknown }).name === 'string') {
    return (e as { name: string }).name
  }
  return 'UnknownError'
}
