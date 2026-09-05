import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

type CognitoEvent = {
  request: { userAttributes: Record<string, string> }
  response?: { claimsOverrideDetails?: { claimsToAddOrOverride?: Record<string, string> } }
}

/**
 * One attribute, crossing from DynamoDB into a token.
 *
 * DynamoDB describes an item as `unknown` per attribute and Cognito accepts
 * only strings, so the gap is real and every claim has to cross it. Typing the
 * record as `Record<string, unknown>` was right — it just turned a coercion
 * that had been happening implicitly under `any` into one the compiler makes
 * you write down.
 *
 * Absent and null both become '' rather than the string "undefined", which is
 * what a bare `String(value)` would have put into a live token.
 */
const claim = (value: unknown): string =>
  typeof value === 'string' ? value : value == null ? '' : String(value)

export const handler = async (event: CognitoEvent) => {
  const { request } = event;
  const cognitoSub = request.userAttributes.sub;

  try {
    let record: Record<string, unknown> | undefined;

    const emailSubQuery = await docClient.send(new QueryCommand({
      TableName: TABLE_NAME,
      IndexName: 'emailSub-index',
      KeyConditionExpression: 'emailSub = :sub',
      ExpressionAttributeValues: { ':sub': cognitoSub },
      Limit: 1,
    }));

    if (emailSubQuery.Items?.[0]) {
      record = emailSubQuery.Items[0];
    } else {
      const googleSubQuery = await docClient.send(new QueryCommand({
        TableName: TABLE_NAME,
        IndexName: 'googleSub-index',
        KeyConditionExpression: 'googleSub = :sub',
        ExpressionAttributeValues: { ':sub': cognitoSub },
        Limit: 1,
      }));

      if (googleSubQuery.Items?.[0]) {
        record = googleSubQuery.Items[0];
      }
    }

    // Never throw from here. This trigger runs on EVERY token issuance, so a
    // missing row or a DynamoDB blip used to mean nobody could sign in at all —
    // a full authentication outage caused by a lookup that is only enriching
    // claims. Now that one Cognito user is one person, `sub` is a perfectly good
    // identity and the mapping is an optimisation, not a prerequisite.
    if (!record) {
      console.warn('USER_MAPPING_NOT_FOUND_FALLBACK', JSON.stringify({ cognitoSub }));
      event.response = {
        claimsOverrideDetails: {
          claimsToAddOrOverride: {
            'custom:appUserId': cognitoSub,
            'custom:email': (request.userAttributes.email as string) || '',
            'custom:displayName': (request.userAttributes.preferred_username as string) || '',
          },
        },
      };
      return event;
    }

    event.response = {
      claimsOverrideDetails: {
        claimsToAddOrOverride: {
          // Falls back to `cognitoSub` for the same reason the not-found branch
          // above uses it: a row that exists but carries no appUserId would
          // otherwise put an EMPTY identity claim into a valid token, which is
          // worse than the sub that already identifies this person.
          'custom:appUserId': claim(record.appUserId) || cognitoSub,
          'custom:emailSub': claim(record.emailSub),
          'custom:googleSub': claim(record.googleSub),
          // Since PreSignUp started linking instead of creating a second user,
          // `googleSub` is no longer written on every path — this flag is.
          'custom:googleLinked': record.googleLinked || record.googleSub ? 'true' : 'false',
          'custom:displayName': claim(record.displayName),
          'custom:email': claim(record.email),
        },
      },
    };

    console.log('INJECTED_CLAIMS', JSON.stringify({
      cognitoSub,
      appUserId: record.appUserId,
      hasEmailSub: !!record.emailSub,
      hasGoogleSub: !!record.googleSub,
    }));
  } catch (error: unknown) {
    // Same rule as above: a DynamoDB failure degrades the claims, it does not
    // lock the user out. `sub` still identifies them.
    console.error('PRETOKEN_DEGRADED', JSON.stringify({ cognitoSub, error: error instanceof Error ? error.message : String(error) }));
    event.response = {
      claimsOverrideDetails: {
        claimsToAddOrOverride: {
          'custom:appUserId': cognitoSub,
          'custom:email': (request.userAttributes.email as string) || '',
        },
      },
    };
  }

  return event;
};
