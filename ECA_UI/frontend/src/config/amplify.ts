export let isAmplifyConfigured = false

export async function initializeAmplify(): Promise<void> {
  // TODO: uncomment this when deploy in production
  // try {
  //   const { Amplify } = await import('aws-amplify')
  //   const outputs = await import(/* @vite-ignore */ '../../amplify_outputs.json')
  //   if (!outputs.default || Object.keys(outputs.default).length === 0) {
  //     throw new Error('Amplify outputs are empty')
  //   }
  //   Amplify.configure(outputs.default)
  //   isAmplifyConfigured = true
  //   console.log('[Auth] Amplify configured successfully')
  // } catch {
  //   console.warn(
  //     '%c⚠ Amplify not configured — running in demo mode',
  //     'color: #f59e0b; font-weight: bold'
  //   )
  //   console.warn('[Auth] Run `npx ampx sandbox` to enable authentication')
  // }
}
