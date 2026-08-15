type ClerkAuthProvider = {
  getToken: () => Promise<string | null>
  userId: string | null
}

let provider: ClerkAuthProvider | null = null

export function setClerkAuthProvider(next: ClerkAuthProvider | null) {
  provider = next
}

export async function clerkAuthHeader(): Promise<Record<string, string>> {
  try {
    const token = await provider?.getToken()
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

export function clerkUserId(): string | null {
  return provider?.userId ?? null
}
