import { ALLOWED_ORIGINS, WEB_DEV_ORIGIN } from '../../shared/origins';

/**
 * CORS headers for one request, reflecting the caller's origin when it is
 * allowed.
 *
 * A single hardcoded origin cannot serve web and mobile at once: the browser
 * compares `Access-Control-Allow-Origin` against the *actual* Origin, so a
 * response naming the Vite dev server is rejected on device. Reflecting the
 * request origin — only after checking the allowlist — is what makes one API
 * serve both.
 *
 * The header is omitted entirely for an origin that is not allowed. Echoing an
 * arbitrary origin back would turn the allowlist into decoration.
 */
type CorsEvent = { headers?: Record<string, string | undefined> }

export function corsHeadersFor(event: CorsEvent): Record<string, string> {
  const headers = (event as { headers?: Record<string, string | undefined> })?.headers ?? {};
  // API Gateway does not normalise header casing.
  const origin: string | undefined = headers.origin ?? headers.Origin;

  const base: Record<string, string> = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
    // Two different origins can receive different answers to the same URL, so
    // caches must key on it.
    Vary: 'Origin',
  };

  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    return { ...base, 'Access-Control-Allow-Origin': origin };
  }

  // No origin header at all means a non-browser caller (curl, a native HTTP
  // client, a health check) — CORS does not apply, so answering normally is
  // correct. Fall back to the dev origin purely to keep the shape stable.
  if (!origin) {
    return { ...base, 'Access-Control-Allow-Origin': WEB_DEV_ORIGIN };
  }

  // Log what it was compared against, not just what was rejected. Without the
  // allow-list the CloudWatch line says "rejected" and nothing else, while the
  // browser reports a CORS failure on a 200 — neither end tells you the list is
  // short because WEB_APP_ORIGIN was unset when the backend was synthesised.
  console.warn('CORS_ORIGIN_REJECTED', JSON.stringify({
    origin,
    allowed: ALLOWED_ORIGINS,
    hint: 'set WEB_APP_ORIGIN on the Amplify branch and REBUILD — the list is baked in at synth time',
  }));
  return base;
}
