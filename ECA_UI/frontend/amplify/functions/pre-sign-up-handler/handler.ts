import {
  AdminCreateUserCommand,
  AdminLinkProviderForUserCommand,
  AdminSetUserPasswordCommand,
  CognitoIdentityProviderClient,
  ListUsersCommand,
  type UserType,
} from '@aws-sdk/client-cognito-identity-provider';

import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand, UpdateCommand } from '@aws-sdk/lib-dynamodb';

const cognito = new CognitoIdentityProviderClient({});
const docClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

/**
 * PreSignUp is the only place that can refuse a federated identity *before*
 * Cognito commits it. Everything downstream — PostConfirmation, the frontend
 * mismatch check — runs after the user already exists, which is why picking the
 * wrong Google account used to leave a real account behind.
 *
 * The model: exactly one Cognito user per human, always the native one. A Google
 * identity is attached to it with AdminLinkProviderForUser rather than becoming
 * a second user. Linking has to happen here: once Cognito has created the
 * federated user, linking it to a native user fails with AliasExistsException
 * and the only way back is to delete the account.
 */

/** Google's `sub`, taken from the `Google_<sub>` username Cognito synthesises. */
function externalSubFrom(userName: string): string | null {
  const separator = userName.indexOf('_');
  if (separator <= 0 || separator === userName.length - 1) return null;
  return userName.slice(separator + 1);
}

function isFederated(user: UserType): boolean {
  return user.Attributes?.some((a) => a.Name === 'identities' && !!a.Value) ?? false;
}

/**
 * Native, confirmed users holding this email.
 *
 * Both filters are security-critical, not tidiness:
 * - federated users are excluded because linking a Google identity onto another
 *   provider's user is meaningless;
 * - UNCONFIRMED users are excluded because anyone can start a sign-up with
 *   somebody else's address. Linking a verified Google identity onto that
 *   unverified shell would hand the squatter — who chose its password — a way
 *   into the real owner's account.
 */
async function findLinkableNativeUsers(userPoolId: string, email: string): Promise<UserType[]> {
  // ListUsers' filter syntax is quoted; an embedded quote would let a crafted
  // address change the predicate.
  if (email.includes('"') || email.includes('\\')) {
    throw new Error('INVALID_EMAIL');
  }

  const result = await cognito.send(new ListUsersCommand({
    UserPoolId: userPoolId,
    Filter: `email = "${email}"`,
    Limit: 60,
  }));

  return (result.Users ?? []).filter(
    (u) => u.UserStatus === 'CONFIRMED' && !isFederated(u),
  );
}

/** A password nobody knows and nobody needs: it exists only so the placeholder
 *  account reaches CONFIRMED instead of sitting in FORCE_CHANGE_PASSWORD, where
 *  it could neither sign in nor be linked to. The user sets a real one later
 *  through set-password or the forgot-password flow. */
function unusablePassword(): string {
  return `Aa1!${crypto.randomUUID()}${crypto.randomUUID()}`;
}

/** Create the native user that a Google-first sign-up will be anchored to. */
async function createNativeAnchor(
  userPoolId: string,
  email: string,
  attributes: Record<string, string>,
): Promise<string> {
  const displayName =
    [attributes.given_name, attributes.family_name].filter(Boolean).join(' ').trim() ||
    email.split('@')[0];

  try {
    await cognito.send(new AdminCreateUserCommand({
      UserPoolId: userPoolId,
      Username: email,
      MessageAction: 'SUPPRESS',
      UserAttributes: [
        { Name: 'email', Value: email },
        { Name: 'email_verified', Value: 'true' },
        { Name: 'preferred_username', Value: displayName },
      ],
    }));
  } catch (error: unknown) {
    // Two first sign-ins racing each other. The loser just adopts the winner's
    // user, which is the same outcome — no lock needed.
    if ((error as { name?: string }).name !== 'UsernameExistsException') throw error;
    console.log('ANCHOR_ALREADY_EXISTS', JSON.stringify({ email }));
    return email;
  }

  await cognito.send(new AdminSetUserPasswordCommand({
    UserPoolId: userPoolId,
    Username: email,
    Password: unusablePassword(),
    Permanent: true,
  }));

  console.log('ANCHOR_CREATED', JSON.stringify({ email }));
  return email;
}

async function linkGoogleTo(
  userPoolId: string,
  destinationUsername: string,
  googleSub: string,
): Promise<void> {
  try {
    await cognito.send(new AdminLinkProviderForUserCommand({
      UserPoolId: userPoolId,
      DestinationUser: {
        ProviderName: 'Cognito',
        ProviderAttributeValue: destinationUsername,
      },
      SourceUser: {
        ProviderName: 'Google',
        ProviderAttributeName: 'Cognito_Subject',
        ProviderAttributeValue: googleSub,
      },
    }));
  } catch (error: unknown) {
    // Already linked — a retried redirect, or the other side of the race above.
    if ((error as { name?: string }).name === 'InvalidParameterException' || (error as { name?: string }).name === 'AliasExistsException') {
      console.log('ALREADY_LINKED', JSON.stringify({ destinationUsername }));
      return;
    }
    throw error;
  }
}

/**
 * Keep the UserMappings row honest about Google being available.
 *
 * PostConfirmation cannot do this any more: an anchor made with AdminCreateUser
 * is an admin action, not a sign-up, so no PostConfirmation fires — and for a
 * linked sign-in Cognito resolves to the existing user without one either.
 * Without this write, `lookup-email` would report `hasGoogle: false` and send
 * a returning Google user to the create-account page.
 *
 * Best-effort on purpose: this table drives a UX hint, and pre-token-generation
 * now falls back to `sub`. Failing the whole sign-in over it would be worse than
 * a stale hint.
 */
async function recordGoogleAvailable(email: string, displayName: string): Promise<void> {
  if (!TABLE_NAME) return;
  try {
    const found = await docClient.send(new QueryCommand({
      TableName: TABLE_NAME,
      IndexName: 'email-index',
      KeyConditionExpression: 'email = :email',
      ExpressionAttributeValues: { ':email': email },
      Limit: 1,
    }));

    const appUserId = found.Items?.[0]?.appUserId ?? crypto.randomUUID();

    await docClient.send(new UpdateCommand({
      TableName: TABLE_NAME,
      Key: { appUserId },
      UpdateExpression:
        'SET email = :email, googleLinked = :true, displayName = if_not_exists(displayName, :dn), createdAt = if_not_exists(createdAt, :now)',
      ExpressionAttributeValues: {
        ':email': email,
        ':true': true,
        ':dn': displayName,
        ':now': new Date().toISOString(),
      },
    }));
  } catch (error: unknown) {
    console.warn('MAPPING_WRITE_SKIPPED', JSON.stringify({ email, error: error instanceof Error ? error.message : String(error) }));
  }
}

type CognitoEvent = {
  triggerSource: string
  request: { userAttributes: Record<string, string> }
  response: { autoConfirmUser?: boolean; autoVerifyEmail?: boolean }
  userName: string
  userPoolId: string
}

export const handler = async (event: CognitoEvent) => {
  const { triggerSource, request, userName, userPoolId } = event;

  if (triggerSource === 'PreSignUp_ExternalProvider') {
    const email: string | undefined = request.userAttributes.email;
    const emailVerified = request.userAttributes.email_verified === 'true';

    if (!email) {
      console.log('EMAIL_MISSING', JSON.stringify({ userName }));
      throw new Error('EMAIL_REQUIRED');
    }

    // Non-negotiable. Without a verified address from the provider, anyone able
    // to assert an arbitrary email at the IdP could claim another user's
    // account. Google verifies; a future provider might not.
    if (!emailVerified) {
      console.log('EMAIL_NOT_VERIFIED', JSON.stringify({ email }));
      throw new Error('EMAIL_NOT_VERIFIED');
    }

    const googleSub = externalSubFrom(userName);
    if (!googleSub) {
      console.error('CANNOT_PARSE_EXTERNAL_SUB', JSON.stringify({ userName }));
      throw new Error('INVALID_EXTERNAL_USERNAME');
    }

    const candidates = await findLinkableNativeUsers(userPoolId, email);

    if (candidates.length > 1) {
      // Should be impossible — email is an alias and Cognito enforces it. If it
      // happens the pool is already inconsistent, so refuse rather than guess
      // which account to hand the Google identity to.
      console.error('AMBIGUOUS_NATIVE_USERS', JSON.stringify({ email, count: candidates.length }));
      throw new Error('AMBIGUOUS_ACCOUNT');
    }

    const destination =
      candidates[0]?.Username ??
      (await createNativeAnchor(userPoolId, email, request.userAttributes));

    await linkGoogleTo(userPoolId, destination, googleSub);

    const displayName =
      [request.userAttributes.given_name, request.userAttributes.family_name]
        .filter(Boolean).join(' ').trim() || email.split('@')[0];
    await recordGoogleAvailable(email, displayName);

    console.log('GOOGLE_LINKED', JSON.stringify({
      email,
      destination,
      anchored: candidates.length === 0,
    }));

    event.response.autoConfirmUser = true;
    event.response.autoVerifyEmail = true;
    return event;
  }

  // Native sign-up. Nothing to decide: Cognito's own email alias already
  // rejects a duplicate address, and a Google identity for the same person is
  // attached by the branch above rather than by a second account.
  return event;
};
