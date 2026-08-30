import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, TransactWriteCommand, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;
const LOCKS_TABLE_NAME = process.env.EMAIL_LOCKS_TABLE_NAME;

type CognitoEvent = {
  request: { userAttributes: Record<string, string> }
}

export const handler = async (event: CognitoEvent) => {
  const { request } = event;
  const email = request.userAttributes.email;
  const cognitoSub = request.userAttributes.sub;
  const isGoogle = !!request.userAttributes.identities;

  if (!email) {
    return event;
  }

  const emailPrefix = email.split('@')[0];

  let displayName: string;
  if (isGoogle) {
    const givenName = request.userAttributes['given_name'] || '';
    const familyName = request.userAttributes['family_name'] || '';
    displayName = (givenName + ' ' + familyName).trim() || emailPrefix;
  } else {
    displayName = (request.userAttributes['preferred_username'] || '').trim() || emailPrefix;
  }

  let existing: Record<string, unknown> | undefined;

  try {
    const queryResponse = await docClient.send(new QueryCommand({
      TableName: TABLE_NAME,
      IndexName: 'email-index',
      KeyConditionExpression: 'email = :email',
      ExpressionAttributeValues: { ':email': email },
      Limit: 1,
    }));

    existing = queryResponse.Items?.[0];

    if (!existing) {
      const appUserId = crypto.randomUUID();
      const now = new Date().toISOString();
      const ttl = Math.floor(Date.now() / 1000) + 86400;

      const item: Record<string, string> = {
        appUserId,
        email,
        displayName,
        createdAt: now,
      };
      if (isGoogle) {
        item.googleSub = cognitoSub;
      } else {
        item.emailSub = cognitoSub;
      }

      await docClient.send(new TransactWriteCommand({
        TransactItems: [
          {
            Put: {
              TableName: LOCKS_TABLE_NAME,
              Item: { email, createdAt: now, ttl },
              ConditionExpression: 'attribute_not_exists(email)',
            },
          },
          {
            Put: {
              TableName: TABLE_NAME,
              Item: item,
              ConditionExpression: 'attribute_not_exists(appUserId)',
            },
          },
        ],
      }));

      console.log('ACCOUNT_CREATED', JSON.stringify({ appUserId, email, type: isGoogle ? 'google' : 'email' }));
      return event;
    }

    const subField = isGoogle ? 'googleSub' : 'emailSub';
    await docClient.send(new TransactWriteCommand({
      TransactItems: [
        {
          Update: {
            TableName: TABLE_NAME,
            Key: { appUserId: existing.appUserId },
            UpdateExpression: `SET ${subField} = :sub`,
            ConditionExpression: isGoogle
              ? 'attribute_not_exists(googleSub)'
              : 'attribute_not_exists(emailSub)',
            ExpressionAttributeValues: { ':sub': cognitoSub },
          },
        },
      ],
    }));

    console.log('ACCOUNT_MAPPED_EXISTING', JSON.stringify({
      appUserId: existing.appUserId, email, subField, cognitoSub,
    }));
  } catch (error: unknown) {
    if ((error as { name?: string }).name === 'TransactionCanceledException') {
      if (!existing) {
        const retryQuery = await docClient.send(new QueryCommand({
          TableName: TABLE_NAME,
          IndexName: 'email-index',
          KeyConditionExpression: 'email = :email',
          ExpressionAttributeValues: { ':email': email },
          Limit: 1,
        }));

        const retryExisting = retryQuery.Items?.[0];
        if (retryExisting) {
          const subField = isGoogle ? 'googleSub' : 'emailSub';
          await docClient.send(new TransactWriteCommand({
            TransactItems: [
              {
                Update: {
                  TableName: TABLE_NAME,
                  Key: { appUserId: retryExisting.appUserId },
                  UpdateExpression: `SET ${subField} = :sub`,
                  ConditionExpression: isGoogle
                    ? 'attribute_not_exists(googleSub)'
                    : 'attribute_not_exists(emailSub)',
                  ExpressionAttributeValues: { ':sub': cognitoSub },
                },
              },
            ],
          }));

          console.log('ACCOUNT_MAPPED_EXISTING_RETRY', JSON.stringify({
            appUserId: retryExisting.appUserId, email, subField, cognitoSub,
          }));
          return event;
        }
      }

      console.warn('PROVIDER_ALREADY_LINKED', JSON.stringify({
        email, cognitoSub, type: isGoogle ? 'google' : 'email',
      }));
      return event;
    }

    console.error('POST_CONFIRMATION_FAILED', JSON.stringify({
      email, cognitoSub, error: error instanceof Error ? error.message : String(error),
    }));
    throw error;
  }

  return event;
};
