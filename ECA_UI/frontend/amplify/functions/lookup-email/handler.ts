import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb'
import { corsHeadersFor } from '../shared/cors'

const client = new DynamoDBClient({})
const docClient = DynamoDBDocumentClient.from(client)

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME

interface ApiEvent {
  queryStringParameters?: { email?: string } | null
  headers?: Record<string, string>
  requestContext?: { authorizer?: { claims?: Record<string, string> } }
}

export const handler = async (event: ApiEvent) => {
  try {
    const email = event.queryStringParameters?.email

    if (!email) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'Missing email parameter' }),
        headers: corsHeadersFor(event),
      }
    }

    const result = await docClient.send(new QueryCommand({
      TableName: TABLE_NAME,
      IndexName: 'email-index',
      KeyConditionExpression: 'email = :email',
      ExpressionAttributeValues: { ':email': email },
      Limit: 1,
    }))

    const record = result.Items?.[0]

    return {
      statusCode: 200,
      body: JSON.stringify({
        hasEmail: !!record?.emailSub,
        // `googleLinked` is the flag pre-sign-up sets when it links or anchors a
        // Google identity. `googleSub` is the pre-single-user field, kept so
        // accounts created before the change still answer correctly.
        hasGoogle: !!record?.googleLinked || !!record?.googleSub,
      }),
      headers: corsHeadersFor(event),
    }
  } catch (error: unknown) {
    console.error('LOOKUP_FAILED', JSON.stringify({ errorMessage: error instanceof Error ? error.message : String(error) }))
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to lookup email' }),
      headers: corsHeadersFor(event),
    }
  }
}
