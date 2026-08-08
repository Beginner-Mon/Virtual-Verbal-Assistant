import {
  AdminLinkProviderForUserCommand,
  CognitoIdentityProviderClient,
} from '@aws-sdk/client-cognito-identity-provider'
import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, UpdateCommand } from '@aws-sdk/lib-dynamodb'
import { JwtRsaVerifier } from 'aws-jwt-verify'
import { corsHeadersFor } from '../shared/cors'

/**
 * Attach a Google identity to the CALLER's existing account — without letting
 * Cognito create anything.
 *
 * Why this endpoint exists at all, when the hosted UI can already link:
 *
 * Linking through `signInWithRedirect` is a *sign-up* as far as Cognito is
 * concerned. Picking the wrong account in Google's chooser therefore reached
 * PreSignUp with a stranger's address, and PreSignUp's job is to make sure every
 * address has an account — so it made one. A user who meant to link `a@` and
 * mis-clicked `b@` ended up with a real `b@` Cognito user plus a UserMappings
 * row, and nothing downstream could undo it: PostConfirmation, the token
 * trigger and the frontend mismatch check all run *after* the user exists.
 *
 * There is no fix for that inside PreSignUp, because PreSignUp cannot tell a
 * mis-click during a link from a genuine first-time Google sign-up. The two
 * requests are identical: same client, same provider, same trigger source, an
 * address with no account behind it. Cognito forwards no application state that
 * would separate them — not the OAuth `state` parameter, not client metadata.
 *
 * So the link flow stops going through Cognito. The browser gets an ID token
 * straight from Google, this endpoint verifies it and compares the address to
 * the caller's own, and only then calls AdminLinkProviderForUser. A mismatch
 * returns 409 having created nothing anywhere — no Cognito user, no DynamoDB
 * row — because no sign-up ever happened.
 *
 * The hosted UI keeps handling actual sign-in, where a brand-new Google user
 * *should* get an account.
 */

const cognito = new CognitoIdentityProviderClient({})
const docClient = DynamoDBDocumentClient.from(new DynamoDBClient({}))

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID

/**
 * Google signs ID tokens with keys it rotates, so verification needs the live
 * JWKS. The verifier caches keys across invocations — building it per request
 * would add a network round-trip to every link.
 *
 * Registered under both spellings of the issuer on purpose. Google's own
 * verification guidance accepts `accounts.google.com` and
 * `https://accounts.google.com`, and it does emit both — pinning one would
 * reject real tokens intermittently, which is the worst kind of auth bug to
 * diagnose.
 */
const GOOGLE_JWKS_URI = 'https://www.googleapis.com/oauth2/v3/certs'

const googleVerifier = GOOGLE_CLIENT_ID
  ? JwtRsaVerifier.create([
      {
        issuer: 'https://accounts.google.com',
        audience: GOOGLE_CLIENT_ID,
        jwksUri: GOOGLE_JWKS_URI,
      },
      {
        issuer: 'accounts.google.com',
        audience: GOOGLE_CLIENT_ID,
        jwksUri: GOOGLE_JWKS_URI,
      },
    ])
  : null

/** Only the parts of the API Gateway proxy event this handler reads. */
interface ApiEvent {
  body?: string | null
  headers?: Record<string, string | undefined>
  requestContext?: { authorizer?: { claims?: Record<string, string> } }
}

function emailsMatch(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase()
}

/** AWS SDK errors are discriminated by `name`, and that is all we branch on. */
function errorName(error: unknown): string {
  return error instanceof Error ? error.name : 'UnknownError'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function reply(event: ApiEvent, statusCode: number, body: Record<string, unknown>) {
  return { statusCode, body: JSON.stringify(body), headers: corsHeadersFor(event) }
}

export const handler = async (event: ApiEvent) => {
  if (!googleVerifier) {
    console.error('GOOGLE_CLIENT_ID_MISSING')
    return reply(event, 500, { message: 'Google linking is not configured' })
  }

  const claims = event.requestContext?.authorizer?.claims
  const userPoolId: string | undefined = claims?.iss?.split('/').pop()
  // `custom:email` is injected by the token trigger; `email` is the standard
  // claim. Either is the caller's own address — this is the value the Google
  // account has to match.
  const callerEmail: string | undefined = claims?.['custom:email'] ?? claims?.email
  // Link onto the exact Cognito user the caller is signed in as. Resolving by
  // email instead would silently pick a different user if the pool ever held
  // two rows for one address.
  const callerUsername: string | undefined = claims?.['cognito:username'] ?? callerEmail
  const appUserId: string | undefined = claims?.['custom:appUserId']

  if (!userPoolId || !callerEmail || !callerUsername) {
    return reply(event, 401, { message: 'Unauthorized' })
  }

  let credential: string | undefined
  try {
    credential = JSON.parse(event.body || '{}').credential
  } catch {
    return reply(event, 400, { message: 'Malformed request body' })
  }
  if (!credential) {
    return reply(event, 400, { message: 'Missing Google credential' })
  }

  let googleEmail: string
  let googleSub: string
  try {
    const payload = await googleVerifier.verify(credential)
    googleEmail = String(payload.email ?? '')
    googleSub = String(payload.sub ?? '')

    // Google normally verifies addresses itself, but the claim is per-token and
    // an unverified one would let anyone assert someone else's address.
    if (payload.email_verified !== true && payload.email_verified !== 'true') {
      console.warn('GOOGLE_EMAIL_NOT_VERIFIED', JSON.stringify({ googleEmail }))
      return reply(event, 400, { code: 'EMAIL_NOT_VERIFIED', message: 'Google has not verified this address' })
    }
  } catch (error) {
    // Expired, wrong audience, forged, replayed from another app — all the same
    // answer. The detail goes to the log, not to the caller.
    console.warn('GOOGLE_TOKEN_REJECTED', JSON.stringify({ error: errorMessage(error) }))
    return reply(event, 401, { code: 'INVALID_CREDENTIAL', message: 'Could not verify the Google account' })
  }

  if (!googleEmail || !googleSub) {
    return reply(event, 400, { code: 'INVALID_CREDENTIAL', message: 'Google returned an incomplete profile' })
  }

  // THE CHECK. Everything above exists so this can happen before any write.
  if (!emailsMatch(googleEmail, callerEmail)) {
    console.log('LINK_REFUSED_MISMATCH', JSON.stringify({ callerEmail, googleEmail }))
    return reply(event, 409, {
      code: 'EMAIL_MISMATCH',
      message: `That Google account is ${googleEmail}, but you are signed in as ${callerEmail}.`,
      googleEmail,
    })
  }

  try {
    await cognito.send(new AdminLinkProviderForUserCommand({
      UserPoolId: userPoolId,
      DestinationUser: {
        ProviderName: 'Cognito',
        ProviderAttributeValue: callerUsername,
      },
      SourceUser: {
        ProviderName: 'Google',
        ProviderAttributeName: 'Cognito_Subject',
        ProviderAttributeValue: googleSub,
      },
    }))
  } catch (error) {
    // Already attached — a double click, or a retry after a timeout. The user
    // asked for Google to be linked and it is, so this is success.
    const name = errorName(error)
    if (name === 'InvalidParameterException' || name === 'AliasExistsException') {
      console.log('ALREADY_LINKED', JSON.stringify({ callerUsername }))
      return reply(event, 200, { linked: true, alreadyLinked: true })
    }
    console.error('LINK_FAILED', JSON.stringify({ callerUsername, error: name }))
    return reply(event, 500, { message: 'Failed to link the Google account' })
  }

  // Best-effort, exactly as in PreSignUp: this row drives a UX hint on the login
  // page. Failing the whole link over a stale hint would be the worse trade —
  // the link itself already succeeded and cannot be rolled back.
  if (TABLE_NAME && appUserId) {
    try {
      await docClient.send(new UpdateCommand({
        TableName: TABLE_NAME,
        Key: { appUserId },
        UpdateExpression: 'SET googleLinked = :true, googleSub = :sub',
        ExpressionAttributeValues: { ':true': true, ':sub': googleSub },
      }))
    } catch (error) {
      console.warn('MAPPING_WRITE_SKIPPED', JSON.stringify({ appUserId, error: errorMessage(error) }))
    }
  }

  console.log('GOOGLE_LINKED_DIRECT', JSON.stringify({ callerEmail, callerUsername }))
  return reply(event, 200, { linked: true })
}
