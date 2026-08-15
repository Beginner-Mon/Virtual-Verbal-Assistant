// Must be the FIRST import: it loads .env before the module-evaluation-time
// consts below (PROD_WEB_ORIGIN, ALLOWED_ORIGINS, OAUTH_*_URLS) are computed.
import './env'

/**
 * Every origin allowed to talk to the auth API, in one place.
 *
 * These values were previously written as the literal string
 * 'http://localhost:5173' in eleven separate places — four CORS preflights, two
 * API Gateway error responses, three Lambda handlers, and the two OAuth redirect
 * lists. A Capacitor build does not run on that origin, so every one of them
 * would have rejected the mobile app, and finding all eleven is exactly the kind
 * of thing that gets half-done.
 */

/** Vite dev server. */
export const WEB_DEV_ORIGIN = 'http://localhost:5173';

/**
 * The origin the mobile app serves itself from.
 *
 * Capacitor would default to `capacitor://localhost` (iOS) or `http://localhost`
 * (Android), and that default is unusable here. Amplify picks which redirect URI
 * to use by matching the browser's own origin, with a substring test on the
 * hostname — on those defaults it happily selects `http://localhost:5173/`, the
 * Vite dev server, because the string "localhost" appears in it. Configure only
 * a custom scheme instead and it throws `invalidRedirectException`, since it
 * accepts nothing but http(s).
 *
 * So the app is served from a real HTTPS origin the project owns
 * (`server.hostname` in capacitor.config.ts). Amplify then resolves the redirect
 * correctly with no patching, and the OS can route the callback back into the
 * app through App Links / Universal Links.
 *
 * Set MOBILE_APP_ORIGIN at deploy time, e.g. https://app.example.com
 */
export const MOBILE_APP_ORIGIN = process.env.MOBILE_APP_ORIGIN;

/**
 * Production web origin.
 *
 * Read from the app root's `.env` file (loaded by ./env.ts) or, failing that,
 * from the process environment — the CI's console variables win over the file.
 * Deliberately not defaulted to anything: a wrong guess here either blocks the
 * real site or silently widens the allowlist.
 */
const PROD_WEB_ORIGIN = process.env.WEB_APP_ORIGIN;

export const ALLOWED_ORIGINS: string[] = [
  WEB_DEV_ORIGIN,
  ...(MOBILE_APP_ORIGIN ? [MOBILE_APP_ORIGIN] : []),
  ...(PROD_WEB_ORIGIN ? [PROD_WEB_ORIGIN] : []),
];

/**
 * OAuth redirect targets registered with Cognito.
 *
 * Every entry is https (or the http dev server) — no custom scheme, because
 * Amplify's web build refuses to select one. The mobile entry is an App Link:
 * an ordinary HTTPS URL that the operating system hands to the installed app
 * instead of the browser.
 *
 * Cognito accepting these is necessary but not sufficient. The same URIs must
 * also be authorised redirect URIs in the Google Cloud console, and the domain
 * must serve the association files that let the OS claim the link:
 *   Android  /.well-known/assetlinks.json
 *   iOS      /.well-known/apple-app-site-association
 */
export const MOBILE_REDIRECT_PATH = '/auth/callback/';

export const OAUTH_CALLBACK_URLS: string[] = [
  `${WEB_DEV_ORIGIN}/`,
  ...(MOBILE_APP_ORIGIN ? [`${MOBILE_APP_ORIGIN}${MOBILE_REDIRECT_PATH}`] : []),
  ...(PROD_WEB_ORIGIN ? [`${PROD_WEB_ORIGIN}/`] : []),
];

/**
 * Sign-out targets. `/login` is here for a specific reason.
 *
 * When a Google sign-in comes back as the wrong account we cannot simply clear
 * local storage and redirect: that leaves Cognito's managed-login session cookie
 * intact on the Cognito domain, so the *next* attempt is silently authorised as
 * that same wrong account without ever asking again. Ending the session properly
 * means bouncing through Cognito's /logout, and `logout_uri` must exactly match
 * one of these registered values.
 */
export const OAUTH_LOGOUT_URLS: string[] = [
  `${WEB_DEV_ORIGIN}/`,
  `${WEB_DEV_ORIGIN}/login`,
  ...(MOBILE_APP_ORIGIN ? [`${MOBILE_APP_ORIGIN}/`, `${MOBILE_APP_ORIGIN}/login`] : []),
  ...(PROD_WEB_ORIGIN ? [`${PROD_WEB_ORIGIN}/`, `${PROD_WEB_ORIGIN}/login`] : []),
];
