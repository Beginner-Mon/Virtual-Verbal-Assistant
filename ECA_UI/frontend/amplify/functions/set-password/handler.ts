import {
  CognitoIdentityProviderClient,
  AdminSetUserPasswordCommand,
  AdminGetUserCommand,
} from '@aws-sdk/client-cognito-identity-provider'
import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, UpdateCommand } from '@aws-sdk/lib-dynamodb'
import { corsHeadersFor } from '../shared/cors'

const cognitoClient = new CognitoIdentityProviderClient({})
const dynamoClient = new DynamoDBClient({})
const docClient = DynamoDBDocumentClient.from(dynamoClient)

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME

export const handler = async (event: any) => {
  try {
    const body = JSON.parse(event.body || '{}')
    const { password } = body

    if (!password || password.length < 8) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'Password must be at least 8 characters' }),
        headers: corsHeadersFor(event),
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
        headers: corsHeadersFor(event),
      }
    }

    // This used to call AdminCreateUser, deliberately minting a SECOND Cognito
    // user for the same person and recording its sub as `emailSub`. That was the
    // main source of duplicate accounts. The native user is now created up front
    // by the PreSignUp trigger — for a Google-first sign-up it is the anchor the
    // Google identity was linked onto — so there is always exactly one user and
    // setting a password is just setting a password on it.
    let nativeSub: string
    try {
      const existing = await cognitoClient.send(new AdminGetUserCommand({
        UserPoolId: userPoolId,
        Username: email,
      }))
      nativeSub = existing.UserAttributes?.find((a) => a.Name === 'sub')?.Value ?? ''
    } catch (e: any) {
      // Only reachable for accounts created before the single-user model, or if
      // PreSignUp failed to anchor. Creating one here would resurrect the
      // duplicate, so fail loudly instead.
      console.error('NATIVE_USER_MISSING', JSON.stringify({ email, error: e.name }))
      return {
        statusCode: 409,
        body: JSON.stringify({
          message: 'This account predates the current sign-in model. Please sign out and sign in again.',
        }),
        headers: corsHeadersFor(event),
      }
    }

    if (!nativeSub) {
      return {
        statusCode: 500,
        body: JSON.stringify({ message: 'Failed to resolve native user' }),
        headers: corsHeadersFor(event),
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
      headers: corsHeadersFor(event),
    }
  } catch (error: any) {
    console.error('SET_PASSWORD_FAILED', JSON.stringify({ errorMessage: error.message }))
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to set password' }),
      headers: corsHeadersFor(event),
    }
  }
}
