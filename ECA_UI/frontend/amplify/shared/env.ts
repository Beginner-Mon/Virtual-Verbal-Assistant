/**
 * Deploy-time env loading for the Amplify backend.
 *
 * The ampx CLI does NOT read .env files itself (verified in @aws-amplify/backend-cli
 * — no dotenv, it only sees process.env). Values like WEB_APP_ORIGIN that the
 * backend reads at deploy time therefore had to be exported into the shell or set
 * in CI, which is easy to forget. This file closes that gap: it loads the app
 * root's .env into process.env before anything else in the backend bundle runs.
 *
 * Rules:
 *  - `process.loadEnvFile` does NOT override variables already present in
 *    process.env, so a CI environment (Amplify Console env vars) always wins
 *    over this file — same convention as agenticRAG's env loader (override=false).
 *  - A missing .env is not an error: sandbox deploys of the web origin
 *    (localhost:5173) need nothing from it.
 *  - Imported as the FIRST import of shared/origins.ts because ES module imports
 *    execute in declaration order, and origins.ts computes its exported consts at
 *    module-evaluation time.
 */
import { existsSync } from 'node:fs'
import { join } from 'node:path'

try {
  const envFile = join(process.cwd(), '.env')
  if (existsSync(envFile)) {
    process.loadEnvFile(envFile)
  }
} catch {
  // .env is optional — a failure here must never block a deploy.
}
