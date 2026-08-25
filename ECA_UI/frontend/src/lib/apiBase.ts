/**
 * The two base URLs this app talks to, and the split between them. One for the
 * API, one for the big binaries — down from four a week ago.
 *
 * There were four until 20-08 — VITE_API_BASE_URL, VITE_CRUD_API_URL,
 * VITE_ASSET_BASE_URL and VITE_AUTH_API_URL — each a separate chance to
 * misconfigure a deployment, and two of them did exactly that in one week. Every
 * backend call now goes through one API Gateway; the CDN keeps only the job a
 * gateway cannot do.
 *
 *   API_GATEWAY   every backend call: /chat, /characters, /sessions, /me/*,
 *                 /health, /tts, /billing/*. The word "later" stood next to
 *                 /chat until 21-08 and is now gone — the agent runs on Lambda
 *                 and the route is an AWS_PROXY integration with
 *                 ResponseTransferMode.STREAM, which is the reason this is a
 *                 REST API and not an HTTP API.
 *   ASSET_BASE    the .vrm model files, 9-17 MB each, from S3 via CloudFront.
 *                 They cannot go through the gateway: API Gateway's buffered
 *                 payload limit is 10 MB, and its data transfer has no free tier
 *                 while CloudFront's first terabyte each month is free.
 *
 * Missing configuration warns rather than throws. This used to throw at module
 * load, and because Vite bakes env vars in at BUILD time a CI build with none
 * shipped a blank page whose only symptom was a message telling you to create a
 * file on a machine that was not the one serving it. A bad value now costs a
 * failed request, not a dead app — and vite.config.ts fails the *build* when a
 * required variable is missing, which is where that belongs.
 */

function resolve(name: string, value: string | undefined, fallback: string): string {
  // trim() first, and it is not defensive programming for its own sake: pasting a
  // URL into the Amplify Console on 20-08 left a leading space, and the resulting
  // value was " https://…" — which fetch() rejects with a message about the URL
  // being invalid rather than about the whitespace. Nothing upstream trims it.
  const cleaned = value?.trim()
  if (!cleaned) {
    console.warn(
      `[config] ${name} is not set — falling back to ${fallback}. Vite reads this ` +
        'at BUILD time: set it in .env.local for local development, or as an ' +
        'Amplify Console environment variable for a deployed build.'
    )
    return fallback
  }
  // A trailing slash doubles up with the leading slash on every path and yields
  // //characters, which API Gateway routes to nothing.
  return cleaned.replace(/\/+$/, '')
}

/** Every backend call. Output `RestApiUrl` of VvaRestApiStack. */
export const API_GATEWAY: string = resolve(
  'VITE_API_GATEWAY_URL',
  import.meta.env.VITE_API_GATEWAY_URL as string | undefined,
  'http://localhost:8000'
)

/** The .vrm files only. Output `AssetBaseUrl` of VvaAssetStack. */
export const ASSET_BASE: string = resolve(
  'VITE_ASSET_BASE_URL',
  import.meta.env.VITE_ASSET_BASE_URL as string | undefined,
  'http://localhost:8000'
)
