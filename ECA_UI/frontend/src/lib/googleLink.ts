import { fetchAuthSession } from 'aws-amplify/auth'
import { customOutputs } from '../config/amplify'

/**
 * Linking Google to an account the user is already signed into.
 *
 * This does NOT go through Cognito's hosted UI. Going that way makes the link a
 * *sign-up*: picking the wrong account in Google's chooser reached the PreSignUp
 * trigger with a stranger's address, and PreSignUp — unable to tell a mis-click
 * from a genuine first-time Google user — created a real account for it. See
 * amplify/functions/link-google/handler.ts.
 *
 * Here the browser asks Google directly for an ID token and hands it to an
 * authenticated endpoint, which compares the address to the caller's own before
 * writing anything. A wrong pick is a 409 and nothing else.
 *
 * A second, smaller win: `login_hint` actually works on this path. Cognito is
 * documented as unable to forward it to Google, so the hosted UI could never
 * pre-select the account — Google Identity Services takes it directly.
 */

const GSI_SRC = 'https://accounts.google.com/gsi/client'

export const GOOGLE_CLIENT_ID: string | undefined =
  (customOutputs?.googleClientId as string | undefined) || undefined

interface GsiCredentialResponse {
  credential: string
}

interface GsiIdApi {
  initialize(config: {
    client_id: string
    callback: (response: GsiCredentialResponse) => void
    login_hint?: string
    auto_select?: boolean
    cancel_on_tap_outside?: boolean
    use_fedcm_for_prompt?: boolean
  }): void
  renderButton(parent: HTMLElement, options: Record<string, unknown>): void
  disableAutoSelect(): void
}

declare global {
  interface Window {
    google?: { accounts?: { id?: GsiIdApi } }
  }
}

let scriptPromise: Promise<GsiIdApi> | null = null

/** Load the GSI script once per page, whoever asks first. */
export function loadGoogleIdentityServices(): Promise<GsiIdApi> {
  if (scriptPromise) return scriptPromise

  scriptPromise = new Promise<GsiIdApi>((resolve, reject) => {
    const ready = () => {
      const api = window.google?.accounts?.id
      if (api) resolve(api)
      else reject(new Error('Google Identity Services loaded without an id API'))
    }

    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`)
    if (existing) {
      if (window.google?.accounts?.id) ready()
      else existing.addEventListener('load', ready, { once: true })
      return
    }

    const script = document.createElement('script')
    script.src = GSI_SRC
    script.async = true
    script.defer = true
    script.addEventListener('load', ready, { once: true })
    script.addEventListener('error', () => {
      // Let a later attempt retry: a network blip should not disable linking
      // for the rest of the session.
      scriptPromise = null
      reject(new Error('Could not reach Google. Check your connection and try again.'))
    }, { once: true })
    document.head.appendChild(script)
  })

  return scriptPromise
}

export interface MountLinkButtonOptions {
  /** The signed-in user's address. Pre-selects it in Google's chooser. */
  loginHint?: string
  onCredential: (credential: string) => void
  onError: (message: string) => void
}

/**
 * Render Google's own button into `container`.
 *
 * A rendered button rather than One Tap (`prompt()`) on purpose: One Tap is
 * suppressed by browser settings, by FedCM, and after a previous dismissal, and
 * it fails silently when it is. A button the user clicks always produces a
 * credential.
 */
export async function mountGoogleLinkButton(
  container: HTMLElement,
  { loginHint, onCredential, onError }: MountLinkButtonOptions,
): Promise<void> {
  if (!GOOGLE_CLIENT_ID) {
    onError('Google sign-in is not configured (googleClientId is missing from Amplify outputs).')
    return
  }

  let api: GsiIdApi
  try {
    api = await loadGoogleIdentityServices()
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Could not load Google sign-in')
    return
  }

  api.initialize({
    client_id: GOOGLE_CLIENT_ID,
    login_hint: loginHint,
    // Never link an account the user did not just choose in this session.
    auto_select: false,
    callback: (response) => onCredential(response.credential),
  })
  api.disableAutoSelect()

  container.replaceChildren()
  api.renderButton(container, {
    type: 'standard',
    theme: 'outline',
    size: 'medium',
    text: 'continue_with',
  })
}

export type LinkResult =
  | { ok: true; alreadyLinked: boolean }
  | { ok: false; code: 'EMAIL_MISMATCH' | 'INVALID_CREDENTIAL' | 'UNKNOWN'; message: string }

/** Send the Google credential to the backend, which decides whether to link. */
export async function linkGoogleAccount(credential: string): Promise<LinkResult> {
  const apiUrl = customOutputs?.authApiUrl as string | undefined
  const session = await fetchAuthSession()
  const token = session.tokens?.idToken?.toString()

  if (!apiUrl || !token) {
    return { ok: false, code: 'UNKNOWN', message: 'Missing API configuration' }
  }

  const res = await fetch(`${apiUrl}api/user/link-google`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: token },
    body: JSON.stringify({ credential }),
  })

  const body = await res.json().catch(() => ({}))

  if (res.ok) {
    return { ok: true, alreadyLinked: !!body.alreadyLinked }
  }
  return {
    ok: false,
    code: body.code ?? 'UNKNOWN',
    message: body.message ?? 'Could not link the Google account',
  }
}
