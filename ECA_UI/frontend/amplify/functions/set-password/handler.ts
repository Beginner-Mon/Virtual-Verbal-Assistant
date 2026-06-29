import {
  CognitoIdentityProviderClient,
  AdminCreateUserCommand,
  AdminSetUserPasswordCommand,
  AdminGetUserCommand,
} from '@aws-sdk/client-cognito-identity-provider'
import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, UpdateCommand } from '@aws-sdk/lib-dynamodb'

const cognitoClient = new CognitoIdentityProviderClient({})
const dynamoClient = new DynamoDBClient({})
const docClient = DynamoDBDocumentClient.from(dynamoClient)

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME
const corsHeaders = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
}

export const handler = async (event: any) => {
  try {
    const body = JSON.parse(event.body || '{}')
    const { password } = body

    if (!password || password.length < 8) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'Password must be at least 8 characters' }),
        headers: corsHeaders,
      }
    }

    const claims = event.requestContext.authorizer?.claims
    const userPoolId = claims?.iss?.split('/').pop()
    const email = claims?.['custom:email']
    const appUserId = claims?.['custom:appUserId']

    if (!userPoolId || !email || !appUserId) {
      return {
        statusCode: 401,
        body: JSON.stringify({ message: 'Unauthorized' }),
        headers: corsHeaders,
      }
    }

    const displayName = claims?.['custom:displayName'] || ''

    let nativeSub: string

    try {
      const result = await cognitoClient.send(new AdminCreateUserCommand({
        UserPoolId: userPoolId,
        Username: email,
        TemporaryPassword: password,
        MessageAction: 'SUPPRESS',
        UserAttributes: [
          { Name: 'email', Value: email },
          { Name: 'email_verified', Value: 'true' },
          { Name: 'preferred_username', Value: displayName },
        ],
      }))
      nativeSub = result.User?.Attributes?.find(
        (a) => a.Name === 'sub'
      )?.Value!
    } catch (e: any) {
      if (e.name !== 'UsernameExistsException') throw e

      const existing = await cognitoClient.send(new AdminGetUserCommand({
        UserPoolId: userPoolId,
        Username: email,
      }))
      nativeSub = existing.UserAttributes?.find(
        (a) => a.Name === 'sub'
      )?.Value!
    }

    if (!nativeSub) {
      return {
        statusCode: 500,
        body: JSON.stringify({ message: 'Failed to resolve native user' }),
        headers: corsHeaders,
      }
    }

    await cognitoClient.send(new AdminSetUserPasswordCommand({
      UserPoolId: userPoolId,
      Username: email,
      Password: password,
      Permanent: true,
    }))

    await docClient.send(new UpdateCommand({
      TableName: TABLE_NAME,
      Key: { appUserId },
      UpdateExpression: 'SET emailSub = :sub',
      ConditionExpression: 'attribute_not_exists(emailSub)',
      ExpressionAttributeValues: { ':sub': nativeSub },
    }))

    console.log('PASSWORD_SETUP_COMPLETE', JSON.stringify({ appUserId, email, nativeSub }))
    return {
      statusCode: 200,
      body: JSON.stringify({ message: 'Password set successfully' }),
      headers: corsHeaders,
    }
  } catch (error: any) {
    console.error('SET_PASSWORD_FAILED', JSON.stringify({ errorMessage: error.message }))
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to set password' }),
      headers: corsHeaders,
    }
  }
}
