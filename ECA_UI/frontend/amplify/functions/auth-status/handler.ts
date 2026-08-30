import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, GetCommand } from '@aws-sdk/lib-dynamodb';
import { corsHeadersFor } from '../shared/cors';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

interface ApiEvent {
  requestContext?: { authorizer?: { claims?: Record<string, string> } }
  headers?: Record<string, string>
}

export const handler = async (event: ApiEvent) => {
  try {
    const claims = event.requestContext?.authorizer?.claims;
    const appUserId = claims?.['custom:appUserId'];

    if (!appUserId) {
      return {
        statusCode: 401,
        body: JSON.stringify({ message: 'Unauthorized' }),
        headers: corsHeadersFor(event),
      };
    }

    const response = await docClient.send(new GetCommand({
      TableName: TABLE_NAME,
      Key: { appUserId },
    }));

    const record = response.Item;

    if (!record) {
      return {
        statusCode: 404,
        body: JSON.stringify({ message: 'User mapping not found' }),
        headers: corsHeadersFor(event),
      };
    }

    return {
      statusCode: 200,
      body: JSON.stringify({
        emailSub: record.emailSub || null,
        googleSub: record.googleSub || null,
        displayName: record.displayName || record.email.split('@')[0],
        email: record.email,
      }),
      headers: corsHeadersFor(event),
    };
  } catch (error: unknown) {
    console.error('AUTH_STATUS_FAILED', JSON.stringify({ errorMessage: error instanceof Error ? error.message : String(error) }));
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to get auth status' }),
      headers: corsHeadersFor(event),
    };
  }
};
