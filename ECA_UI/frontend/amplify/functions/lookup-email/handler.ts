import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb'

const client = new DynamoDBClient({})
const docClient = DynamoDBDocumentClient.from(client)

const TABLE_NAME = process.env.USER_MAPPINGS_TABLE_NAME
const corsHeaders = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
}

export const handler = async (event: any) => {
  try {
    const email = event.queryStringParameters?.email

    if (!email) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'Missing email parameter' }),
        headers: corsHeaders,
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
        hasGoogle: !!record?.googleSub,
      }),
      headers: corsHeaders,
    }
  } catch (error) {
    console.error('LOOKUP_FAILED', JSON.stringify({ errorMessage: (error as Error).message }))
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Failed to lookup email' }),
      headers: corsHeaders,
    }
  }
}
