/**
 * Lightweight email validator — no external deps.
 * Intentionally permissive (HTML5-like) to avoid false-negatives on
 * legitimate addresses like user+tag@sub.example.co.uk.
 * Cognito is still the source of truth; this only prevents
 * obviously-invalid strings from hitting /api/lookup.
 */

export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export function isValidEmail(value: string): boolean {
  const s = value.trim()
  if (!s || s.length > 254) return false
  // must contain exactly one @
  const parts = s.split('@')
  if (parts.length !== 2) return false
  const [local, domain] = parts
  if (!local || !domain) return false
  if (local.length > 64 || domain.length > 253) return false
  if (local.startsWith('.') || local.endsWith('.')) return false
  if (local.includes('..') || domain.includes('..')) return false
  // domain must contain a dot and not start/end with dot or hyphen
  if (!domain.includes('.')) return false
  if (domain.startsWith('.') || domain.endsWith('.')) return false
  if (domain.startsWith('-') || domain.endsWith('-')) return false
  return EMAIL_RE.test(s)
}

export function emailError(value: string): string | null {
  const s = value.trim()
  if (!s) return 'Please enter your email'
  if (!isValidEmail(s)) return 'Please enter a valid email address'
  return null
}
