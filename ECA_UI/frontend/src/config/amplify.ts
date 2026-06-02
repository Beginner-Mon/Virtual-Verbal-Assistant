export let isAmplifyConfigured = false

export async function initializeAmplify(): Promise<void> {
  try {
    const { Amplify } = await import('aws-amplify')
    const outputs = await import(/* @vite-ignore */ '../../amplify_outputs.json')
    Amplify.configure(outputs.default)
    isAmplifyConfigured = true
    console.log('[Auth] Amplify configured successfully')
  } catch {
    console.warn(
      '%c⚠ Amplify not configured — running in demo mode',
      'color: #f59e0b; font-weight: bold'
    )
    console.warn('[Auth] Run `npx ampx sandbox` to enable authentication')
  }
}
