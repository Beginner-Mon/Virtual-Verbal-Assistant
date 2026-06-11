import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

export const handler = async (event: any) => {
  const { request } = event;
  const cognitoSub = request.userAttributes.sub;

  try {
    let appUserId: string | undefined;

    const emailSubQuery = await docClient.send(new QueryCommand({
      TableName: TABLE_NAME,
      IndexName: 'emailSub-index',
      KeyConditionExpression: 'emailSub = :sub',
      ExpressionAttributeValues: { ':sub': cognitoSub },
      Limit: 1,
    }));

    if (emailSubQuery.Items?.[0]) {
      appUserId = emailSubQuery.Items[0].appUserId;
    } else {
      const googleSubQuery = await docClient.send(new QueryCommand({
        TableName: TABLE_NAME,
        IndexName: 'googleSub-index',
        KeyConditionExpression: 'googleSub = :sub',
        ExpressionAttributeValues: { ':sub': cognitoSub },
        Limit: 1,
      }));

      if (googleSubQuery.Items?.[0]) {
        appUserId = googleSubQuery.Items[0].appUserId;
      }
    }

    if (!appUserId) {
      console.error('USER_MAPPING_NOT_FOUND', JSON.stringify({ cognitoSub }));
      throw new Error('USER_MAPPING_NOT_FOUND');
    }

    event.response = {
      claimsOverrideDetails: {
        claimsToAddOrOverride: {
          'custom:appUserId': appUserId,
        },
      },
    };

    console.log('INJECTED_APP_USER_ID', JSON.stringify({ cognitoSub, appUserId }));
  } catch (error: any) {
    if (error.message === 'USER_MAPPING_NOT_FOUND') {
      throw error;
    }
    console.error('PRETOKEN_FAILED', JSON.stringify({ cognitoSub, error: error.message }));
    throw new Error('USER_MAPPING_NOT_FOUND', { cause: error });
  }

  return event;
};
