import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

export const handler = async (event: any) => {
  const { request } = event;
  const cognitoSub = request.userAttributes.sub;

  try {
    let record: Record<string, any> | undefined;

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
          'custom:appUserId': record.appUserId,
          'custom:emailSub': record.emailSub || '',
          'custom:googleSub': record.googleSub || '',
          'custom:displayName': record.displayName || '',
          'custom:email': record.email || '',
        },
      },
    };

    console.log('INJECTED_CLAIMS', JSON.stringify({
      cognitoSub,
      appUserId: record.appUserId,
      hasEmailSub: !!record.emailSub,
      hasGoogleSub: !!record.googleSub,
    }));
  } catch (error: any) {
    // Same rule as above: a DynamoDB failure degrades the claims, it does not
    // lock the user out. `sub` still identifies them.
    console.error('PRETOKEN_DEGRADED', JSON.stringify({ cognitoSub, error: error.message }));
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
