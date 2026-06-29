import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, GetCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;
const corsHeaders = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
};

export const handler = async (event: any) => {
  try {
    const claims = event.requestContext.authorizer?.claims;
    const appUserId = claims?.['custom:appUserId'];

    if (!appUserId) {
      return {
        statusCode: 401,
        body: JSON.stringify({ message: 'Unauthorized' }),
        headers: corsHeaders,
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
        headers: corsHeaders,
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
      headers: corsHeaders,
    };
  } catch (error) {
    console.error('AUTH_STATUS_FAILED', JSON.stringify({ errorMessage: (error as Error).message }));
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to get auth status' }),
      headers: corsHeaders,
    };
  }
};
