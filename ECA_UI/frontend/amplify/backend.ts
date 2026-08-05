import { defineBackend, defineFunction } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { Stack } from 'aws-cdk-lib';
import { CfnUserPool, CfnUserPoolDomain } from 'aws-cdk-lib/aws-cognito';
import { ServicePrincipal, PolicyStatement } from 'aws-cdk-lib/aws-iam';
import { RestApi, LambdaIntegration, CognitoUserPoolsAuthorizer, AuthorizationType, ResponseType } from 'aws-cdk-lib/aws-apigateway';
import { Function } from 'aws-cdk-lib/aws-lambda';

import { Table, AttributeType, BillingMode } from 'aws-cdk-lib/aws-dynamodb';

const preSignUpHandler = defineFunction({
  name: 'pre-sign-up-handler',
  entry: './functions/pre-sign-up-handler/handler.ts',
  timeoutSeconds: 30,
  resourceGroupName: 'auth',
});

const postConfirmationHandler = defineFunction({
  name: 'post-confirmation-handler',
  entry: './functions/post-confirmation/handler.ts',
  timeoutSeconds: 30,
  resourceGroupName: 'auth',
});

const preTokenGenerationHandler = defineFunction({
  name: 'pre-token-generation-handler',
  entry: './functions/pre-token-generation/handler.ts',
  timeoutSeconds: 30,
  resourceGroupName: 'auth',
});

const authStatusHandler = defineFunction({
  name: 'auth-status',
  entry: './functions/auth-status/handler.ts',
  timeoutSeconds: 10,
  resourceGroupName: 'auth',
});

const lookupEmailHandler = defineFunction({
  name: 'lookup-email',
  entry: './functions/lookup-email/handler.ts',
  timeoutSeconds: 5,
});

const setPasswordHandler = defineFunction({
  name: 'set-password',
  entry: './functions/set-password/handler.ts',
  timeoutSeconds: 10,
  resourceGroupName: 'auth',
});

const backend = defineBackend({
  auth,
  preSignUpHandler,
  postConfirmationHandler,
  preTokenGenerationHandler,
  authStatusHandler,
  lookupEmailHandler,
  setPasswordHandler,
});

// --- Cognito User Pool ---
const userPool = backend.auth.resources.userPool;
const cfnUserPool = userPool.node.defaultChild as CfnUserPool;

// --- DynamoDB Tables ---
const userMappingsTable = new Table(backend.stack, 'UserMappings', {
  tableName: 'UserMappings',
  partitionKey: { name: 'appUserId', type: AttributeType.STRING },
  billingMode: BillingMode.PAY_PER_REQUEST,
});

userMappingsTable.addGlobalSecondaryIndex({
  indexName: 'emailSub-index',
  partitionKey: { name: 'emailSub', type: AttributeType.STRING },
});

userMappingsTable.addGlobalSecondaryIndex({
  indexName: 'googleSub-index',
  partitionKey: { name: 'googleSub', type: AttributeType.STRING },
});

userMappingsTable.addGlobalSecondaryIndex({
  indexName: 'email-index',
  partitionKey: { name: 'email', type: AttributeType.STRING },
});

const emailLocksTable = new Table(backend.stack, 'EmailLocks', {
  tableName: 'EmailLocks',
  partitionKey: { name: 'email', type: AttributeType.STRING },
  timeToLiveAttribute: 'ttl',
  billingMode: BillingMode.PAY_PER_REQUEST,
});

// Configure Lambdas
const preSignUpFn = backend.preSignUpHandler.resources.lambda as Function;
const postConfirmationFn = backend.postConfirmationHandler.resources.lambda as Function;
const preTokenGenerationFn = backend.preTokenGenerationHandler.resources.lambda as Function;

// Inject Env Vars
postConfirmationFn.addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);
postConfirmationFn.addEnvironment('EMAIL_LOCKS_TABLE_NAME', emailLocksTable.tableName);
preTokenGenerationFn.addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);
preSignUpFn.addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);
(backend.authStatusHandler.resources.lambda as Function).addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);
(backend.lookupEmailHandler.resources.lambda as Function).addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);
(backend.setPasswordHandler.resources.lambda as Function).addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);

const { region, account } = Stack.of(backend.stack);
const userPoolArnDecoupled = `arn:aws:cognito-idp:${region}:${account}:userpool/*`;

// Grant DynamoDB Permissions
userMappingsTable.grantReadWriteData(postConfirmationFn);
postConfirmationFn.addToRolePolicy(
  new PolicyStatement({
    actions: ['dynamodb:TransactWriteItems'],
    resources: [userMappingsTable.tableArn, emailLocksTable.tableArn],
  }),
);
emailLocksTable.grantReadWriteData(postConfirmationFn);
userMappingsTable.grantReadData(preTokenGenerationFn);
// pre-sign-up now writes too: it records that Google is available for an email,
// which PostConfirmation can no longer do for linked/anchored accounts.
userMappingsTable.grantReadWriteData(preSignUpFn);
userMappingsTable.grantReadData(backend.authStatusHandler.resources.lambda as Function);
userMappingsTable.grantReadData(backend.lookupEmailHandler.resources.lambda as Function);
userMappingsTable.grantReadWriteData(backend.setPasswordHandler.resources.lambda as Function);

// IAM for set-password Lambda.
// AdminCreateUser is deliberately absent: creating a user here is what produced
// the duplicate account. The native user is anchored by PreSignUp instead.
(backend.setPasswordHandler.resources.lambda as Function).addToRolePolicy(
  new PolicyStatement({
    actions: [
      'cognito-idp:AdminSetUserPassword',
      'cognito-idp:AdminGetUser',
    ],
    resources: [userPoolArnDecoupled],
  }),
);

// IAM for pre-sign-up: it now resolves and links accounts instead of blindly
// auto-confirming whatever identity arrives.
preSignUpFn.addToRolePolicy(
  new PolicyStatement({
    actions: [
      'cognito-idp:ListUsers',
      'cognito-idp:AdminLinkProviderForUser',
      'cognito-idp:AdminCreateUser',
      'cognito-idp:AdminSetUserPassword',
    ],
    resources: [userPoolArnDecoupled],
  }),
);

// Attach Triggers
cfnUserPool.addPropertyOverride('LambdaConfig.PreSignUp', preSignUpFn.functionArn);
cfnUserPool.addPropertyOverride('LambdaConfig.PostConfirmation', postConfirmationFn.functionArn);
cfnUserPool.addPropertyOverride('LambdaConfig.PreTokenGeneration', preTokenGenerationFn.functionArn);

const cognitoPrincipal = new ServicePrincipal('cognito-idp.amazonaws.com');
preSignUpFn.addPermission('AllowCognitoPreSignUp', { principal: cognitoPrincipal });
postConfirmationFn.addPermission('AllowCognitoPostConfirmation', { principal: cognitoPrincipal });
preTokenGenerationFn.addPermission('AllowCognitoPreTokenGeneration', { principal: cognitoPrincipal });

// Find the CfnUserPoolDomain that Amplify auto-creates
const cfnDomain = userPool.node
  .tryFindChild('UserPoolDomain')
  ?.node.defaultChild as CfnUserPoolDomain;

if (cfnDomain) {
  cfnDomain.domain = 'eca-us-east-1';
  cfnDomain.addPropertyOverride('ManagedLoginVersion', 2);
}

// Patch missing AttributeDataType on UserPool schema entries
if (cfnUserPool) {
  for (let i = 0; i <= 4; i++) {
    cfnUserPool.addPropertyOverride(`Schema.${i}.AttributeDataType`, 'String');
  }
}

// --- API Gateway ---
const api = new RestApi(backend.stack, 'AuthApi', {
  restApiName: 'VVA Auth API',
});

const authorizer = new CognitoUserPoolsAuthorizer(backend.stack, 'AuthApiAuthorizer', {
  cognitoUserPools: [userPool],
});

// Gateway responses for CORS on auth failures
api.addGatewayResponse('UnauthorizedResponse', {
  type: ResponseType.UNAUTHORIZED,
  responseHeaders: {
    'Access-Control-Allow-Origin': "'http://localhost:5173'",
    'Access-Control-Allow-Headers': "'Content-Type,Authorization'",
  },
});
api.addGatewayResponse('AccessDeniedResponse', {
  type: ResponseType.ACCESS_DENIED,
  responseHeaders: {
    'Access-Control-Allow-Origin': "'http://localhost:5173'",
    'Access-Control-Allow-Headers': "'Content-Type,Authorization'",
  },
});

const apiResource = api.root.addResource('api');

const apiUser = apiResource.addResource('user');
apiUser.addCorsPreflight({
  allowOrigins: ['http://localhost:5173'],
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});

const authStatusRes = apiUser.addResource('auth-status');
authStatusRes.addCorsPreflight({
  allowOrigins: ['http://localhost:5173'],
  allowMethods: ['GET', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});
authStatusRes.addMethod('GET', new LambdaIntegration(backend.authStatusHandler.resources.lambda, {
  proxy: true,
}), {
  authorizer,
  authorizationType: AuthorizationType.COGNITO,
});

const lookupRes = apiResource.addResource('lookup');
lookupRes.addCorsPreflight({
  allowOrigins: ['http://localhost:5173'],
  allowMethods: ['GET', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});
lookupRes.addMethod('GET', new LambdaIntegration(backend.lookupEmailHandler.resources.lambda, {
  proxy: true,
}), {
  authorizationType: AuthorizationType.NONE,
});

const setPasswordRes = apiUser.addResource('set-password');
setPasswordRes.addCorsPreflight({
  allowOrigins: ['http://localhost:5173'],
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});
setPasswordRes.addMethod('POST', new LambdaIntegration(backend.setPasswordHandler.resources.lambda, {
  proxy: true,
}), {
  authorizer,
  authorizationType: AuthorizationType.COGNITO,
});

// Output API URL for frontend
backend.addOutput({
  custom: { authApiUrl: api.url },
});