import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME;

export const handler = async (event: any) => {
  const { triggerSource, request } = event;

  if (triggerSource === 'PreSignUp_ExternalProvider') {
    const email = request.userAttributes.email;
    const emailVerified = request.userAttributes.email_verified === 'true';

    if (!emailVerified) {
      console.log('EMAIL_NOT_VERIFIED', JSON.stringify({ email }));
      throw new Error('EMAIL_NOT_VERIFIED');
    }

    console.log('NEW_FEDERATED_USER_CREATED', JSON.stringify({ email }));
    event.response.autoConfirmUser = true;
    event.response.autoVerifyEmail = true;
    return event;
  }

  if (triggerSource === 'PreSignUp_SignUp') {
    const email = request.userAttributes.email;

    try {
      const queryResponse = await docClient.send(new QueryCommand({
        TableName: TABLE_NAME,
        IndexName: 'email-index',
        KeyConditionExpression: 'email = :email',
        ExpressionAttributeValues: { ':email': email },
        Limit: 1,
      }));

      const existing = queryResponse.Items?.[0];

      if (existing && existing.googleSub) {
        console.log('DUPLICATE_EMAIL_DETECTED', JSON.stringify({ email }));
        throw new Error('EMAIL_EXISTS_USE_GOOGLE');
      }
    } catch (error: any) {
      if (error.message === 'EMAIL_EXISTS_USE_GOOGLE') {
        throw error;
      }
      console.error('PRESIGNUP_FAILED', JSON.stringify({ email, error: error.message }));
    }

    return event;
  }

  return event;
};
