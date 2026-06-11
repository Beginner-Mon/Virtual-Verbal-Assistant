import { CognitoIdentityProviderClient, AdminSetUserPasswordCommand } from '@aws-sdk/client-cognito-identity-provider'

const client = new CognitoIdentityProviderClient({})
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
      return { statusCode: 400, body: JSON.stringify({ message: 'Password must be at least 8 characters' }), headers: corsHeaders }
    }

    const claims = event.requestContext.authorizer?.claims
    const cognitoUsername = claims?.['cognito:username']
    const userPoolId = claims?.iss?.split('/').pop()

    if (!cognitoUsername || !userPoolId) {
      return { statusCode: 401, body: JSON.stringify({ message: 'Unauthorized' }), headers: corsHeaders }
    }

    await client.send(new AdminSetUserPasswordCommand({
      UserPoolId: userPoolId,
      Username: cognitoUsername,
      Password: password,
      Permanent: true,
    }))

    console.log('PASSWORD_CREATED', JSON.stringify({ cognitoSub: cognitoUsername }))

    return { statusCode: 200, body: JSON.stringify({ message: 'Password set successfully' }), headers: corsHeaders }
  } catch (error) {
    console.error('LINKING_FAILED', JSON.stringify({ errorMessage: (error as Error).message }))
    return { statusCode: 500, body: JSON.stringify({ message: 'Failed to set password' }), headers: corsHeaders }
  }
}
