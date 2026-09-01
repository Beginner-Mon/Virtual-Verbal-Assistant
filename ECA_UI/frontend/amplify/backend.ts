import { defineBackend, defineFunction, secret } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { Stack } from 'aws-cdk-lib';
import { CfnUserPool, CfnUserPoolDomain } from 'aws-cdk-lib/aws-cognito';
import { ServicePrincipal, PolicyStatement } from 'aws-cdk-lib/aws-iam';
import { RestApi, LambdaIntegration, CognitoUserPoolsAuthorizer, AuthorizationType, ResponseType } from 'aws-cdk-lib/aws-apigateway';
import { Function } from 'aws-cdk-lib/aws-lambda';

import { Table, AttributeType, BillingMode } from 'aws-cdk-lib/aws-dynamodb';
import { ALLOWED_ORIGINS, OAUTH_CALLBACK_URLS } from './shared/origins';

// Both lists are built from process.env.WEB_APP_ORIGIN at SYNTH time. If that
// variable is not set on the Amplify branch, the deployed app's own origin is in
// neither — the API answers without an Access-Control-Allow-Origin header (the
// browser reports it as a CORS failure on a 200) and Amplify's OAuth flow throws
// InvalidOriginException. Two unrelated-looking errors, one missing variable, so
// the resolved lists go in the build log.
console.log(`[backend] ALLOWED_ORIGINS: ${ALLOWED_ORIGINS.join(', ') || '(none)'}`);
console.log(`[backend] OAUTH_CALLBACK_URLS: ${OAUTH_CALLBACK_URLS.join(', ') || '(none)'}`);
if (!process.env.WEB_APP_ORIGIN) {
  console.warn(
    '[backend] WEB_APP_ORIGIN is NOT set — the deployed origin will be missing ' +
      'from both lists above. Set it on the branch, then rebuild.'
  );
}

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

// Links a Google identity onto the caller's own account WITHOUT going through
// Cognito's sign-up path — see functions/link-google/handler.ts for why the
// hosted UI cannot do this safely. Needs the Google client id as the audience it
// verifies ID tokens against; same secret the provider itself uses, so there is
// one value to rotate.
const linkGoogleHandler = defineFunction({
  name: 'link-google',
  entry: './functions/link-google/handler.ts',
  timeoutSeconds: 10,
  resourceGroupName: 'auth',
  environment: {
    GOOGLE_CLIENT_ID: secret('GOOGLE_CLIENT_ID'),
  },
});

const backend = defineBackend({
  auth,
  preSignUpHandler,
  postConfirmationHandler,
  preTokenGenerationHandler,
  authStatusHandler,
  lookupEmailHandler,
  setPasswordHandler,
  linkGoogleHandler,
});

/**
 * Suffix that makes globally-unique resource names unique PER ENVIRONMENT.
 *
 * Several AWS resource names are unique across an account+region (DynamoDB
 * tables) or across all of AWS (Cognito domain prefixes, S3 buckets). Hardcoding
 * them means exactly one environment can exist. The sandbox created them first,
 * so every pipeline deploy failed CloudFormation's
 * `AWS::EarlyValidation::ResourceExistenceCheck` — before any resource was
 * created, which is why the build log only ever said "Validation failed with
 * 2 error(s)" with no resource-level events behind it. Two hardcoded table
 * names, two errors.
 *
 * Returns null outside CI, so the EXISTING sandbox keeps the names it already
 * has. That is not tidiness: changing `tableName` replaces the table, and
 * replacing it drops every row in it.
 */
function environmentSuffix(): string | null {
  const explicit = process.env.RESOURCE_SUFFIX?.trim();
  if (explicit) return explicit;

  // Both are set by the Amplify build image; absent locally.
  const branch = process.env.AWS_BRANCH;
  const appId = process.env.AWS_APP_ID;
  if (!branch || !appId) return null;

  const slug = branch
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 28);
  // App id suffix keeps two apps building the same branch name apart.
  return `${slug}-${appId.slice(-6).toLowerCase()}`.replace(/-+/g, '-');
}

const ENV_SUFFIX = environmentSuffix();

/** Physical table name for this environment. */
const tableName = (base: string): string =>
  ENV_SUFFIX ? `${base}-${ENV_SUFFIX}` : base;

console.log(
  `[backend] environment suffix: ${ENV_SUFFIX ?? '(none — sandbox, existing names kept)'}`
);

// --- Cognito User Pool ---
const userPool = backend.auth.resources.userPool;
const cfnUserPool = userPool.node.defaultChild as CfnUserPool;

// --- DynamoDB Tables ---
//
// DynamoDB table names are unique per account + region, and CloudFormation's
// early ResourceExistenceCheck fails the whole creation with
// "Validation failed with 2 error(s)" when a template re-uses a name that a
// deployed stack (the sandbox) already holds. The tables below therefore get
// a deployment suffix in CI — see environmentSuffix() above.
const userMappingsTable = new Table(backend.stack, 'UserMappings', {
  // Physical name is per-environment; the Lambdas receive it through
  // USER_MAPPINGS_TABLE_NAME below, so nothing downstream hardcodes it.
  tableName: tableName('UserMappings'),
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
  tableName: tableName('EmailLocks'),
  partitionKey: { name: 'email', type: AttributeType.STRING },
  timeToLiveAttribute: 'ttl',
  billingMode: BillingMode.PAY_PER_REQUEST,
});

console.log(
  `[backend] table names: ${userMappingsTable.tableName}, ${emailLocksTable.tableName}`
);

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
(backend.linkGoogleHandler.resources.lambda as Function).addEnvironment('USER_MAPPINGS_TABLE_NAME', userMappingsTable.tableName);

// Origins are read by shared/origins.ts at module load — esbuild does not
// inline process.env, so that read happens inside the Lambda at runtime, NOT
// on the synth machine. The CI machine saw WEB_APP_ORIGIN (the OAUTH lists on
// the pool prove it), but the live lambdas did not, so their CORS allowlist
// silently omitted the prod origin and every browser call was blocked. Bake
// the origins into the lambdas as env vars.
for (
  const fn of [
    preSignUpFn,
    postConfirmationFn,
    preTokenGenerationFn,
    backend.authStatusHandler.resources.lambda as Function,
    backend.lookupEmailHandler.resources.lambda as Function,
    backend.setPasswordHandler.resources.lambda as Function,
    backend.linkGoogleHandler.resources.lambda as Function,
  ]
) {
  fn.addEnvironment('WEB_APP_ORIGIN', process.env.WEB_APP_ORIGIN ?? '');
  fn.addEnvironment('MOBILE_APP_ORIGIN', process.env.MOBILE_APP_ORIGIN ?? '');
}

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
userMappingsTable.grantReadWriteData(backend.linkGoogleHandler.resources.lambda as Function);

// IAM for link-google. AdminLinkProviderForUser ONLY — deliberately no
// AdminCreateUser: this endpoint attaches an identity to an account that already
// exists, and it must stay structurally incapable of minting one. That is the
// entire difference between it and the hosted-UI link path it replaces.
(backend.linkGoogleHandler.resources.lambda as Function).addToRolePolicy(
  new PolicyStatement({
    actions: ['cognito-idp:AdminLinkProviderForUser'],
    resources: [userPoolArnDecoupled],
  }),
);

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

/**
 * Cognito hosted-UI domain prefix.
 *
 * These are unique across ALL AWS accounts, not just this one — so a hardcoded
 * value means exactly one environment can ever exist. `eca-us-east-1` was
 * hardcoded here, the sandbox took it, and every pipeline deploy after that was
 * going to fail on a name already in use.
 *
 * Resolution order:
 *   1. COGNITO_DOMAIN_PREFIX — explicit, wins. Set it per environment in the
 *      Amplify Console, or in your shell for a second developer sandbox.
 *   2. Derived from the branch + app id in Amplify CI (AWS_BRANCH / AWS_APP_ID
 *      are set by the build image), so each branch gets its own.
 *   3. `eca-us-east-1` — unchanged, so the EXISTING sandbox keeps its domain.
 *      Changing it would replace the domain on a live pool and break every
 *      redirect URI already registered with Google.
 *
 * WHATEVER THIS RESOLVES TO must also appear in:
 *   - the frontend's VITE_COGNITO_DOMAIN (as `<prefix>.auth.<region>.amazoncognito.com`)
 *   - Google Cloud console → Authorized redirect URIs
 *     (`https://<prefix>.auth.<region>.amazoncognito.com/oauth2/idpresponse`)
 * Those are not derivable from here, which is why the prefix is logged below.
 */
function cognitoDomainPrefix(): string {
  const explicit = process.env.COGNITO_DOMAIN_PREFIX?.trim();
  if (explicit) return explicit;
  // Same suffix as the tables, so one environment cannot end up half-scoped.
  // Prefix rules: lowercase alphanumerics and hyphens, no leading or trailing
  // hyphen, 63 chars, and no "aws"/"amazon"/"cognito" — `environmentSuffix`
  // already produces a value that satisfies all of them.
  return ENV_SUFFIX ? `eca-${ENV_SUFFIX}` : 'eca-us-east-1';
}

// Find the CfnUserPoolDomain that Amplify auto-creates
const cfnDomain = userPool.node
  .tryFindChild('UserPoolDomain')
  ?.node.defaultChild as CfnUserPoolDomain;

if (cfnDomain) {
  const prefix = cognitoDomainPrefix();
  // Printed at synth time so the build log records which domain this deployment
  // claimed — the value the Google console and the frontend have to match.
  console.log(`[backend] Cognito hosted-UI domain prefix: ${prefix}`);
  cfnDomain.domain = prefix;
  cfnDomain.addPropertyOverride('ManagedLoginVersion', 2);
}

// Patch missing AttributeDataType on UserPool schema entries.
//
// The loop used to be a hardcoded `i <= 4`, i.e. five entries. `defineAuth`
// declares THREE userAttributes (preferredUsername, givenName, familyName);
// `loginWith.email` becomes UsernameAttributes, not a Schema entry. So indices
// 3 and 4 do not exist, and `addPropertyOverride` on a missing index does not
// no-op — it MATERIALISES the entry, giving CloudFormation two schema members
// that have an AttributeDataType and no Name. Two invalid members, and the
// build fails with "Validation failed with 2 error(s)". The count matching the
// number of declared secrets is a coincidence that cost a day.
//
// Iterating over the real length is the fix, with two caveats worth spelling
// out because the obvious version is wrong:
//
//   - `CfnUserPool.schema` is typed `IResolvable | Array<...>`. When it is an
//     unresolved token, `.length` is `undefined`, and `i < undefined` is false —
//     so `for (let i = 0; i < cfnUserPool.schema.length; i++)` silently applies
//     NO patches and looks like it worked. Hence the explicit Array.isArray.
//   - Zero entries is therefore ambiguous: it means either "no attributes" or
//     "a token we cannot read". It is logged rather than passed over, because
//     the patch exists to satisfy CloudFormation and skipping it entirely brings
//     back whatever it was added to fix.
if (cfnUserPool) {
  const schema = cfnUserPool.schema;
  const entryCount = Array.isArray(schema) ? schema.length : 0;

  console.log(`[backend] user pool schema entries: ${entryCount}`);
  if (entryCount === 0) {
    console.warn(
      '[backend] schema is empty or an unresolved token — AttributeDataType ' +
        'patches were NOT applied. If CloudFormation rejects the schema, this is why.'
    );
  }

  for (let i = 0; i < entryCount; i++) {
    cfnUserPool.addPropertyOverride(`Schema.${i}.AttributeDataType`, 'String');
  }
}

// --- API Gateway ---
const api = new RestApi(backend.stack, 'AuthApi', {
  restApiName: 'ECA Auth API',
});

const authorizer = new CognitoUserPoolsAuthorizer(backend.stack, 'AuthApiAuthorizer', {
  cognitoUserPools: [userPool],
});

// Gateway responses for CORS on auth failures
api.addGatewayResponse('UnauthorizedResponse', {
  type: ResponseType.UNAUTHORIZED,
  responseHeaders: {
    // API Gateway builds these before any integration runs, so the caller's
    // origin cannot be reflected the way the Lambdas do it. '*' is safe for
    // these two: they are 401/403 shells carrying no user data, and the app
    // authenticates with an Authorization header rather than cookies, so no
    // credentialed request is ever widened by it.
    'Access-Control-Allow-Origin': "'*'",
    'Access-Control-Allow-Headers': "'Content-Type,Authorization'",
  },
});
api.addGatewayResponse('AccessDeniedResponse', {
  type: ResponseType.ACCESS_DENIED,
  responseHeaders: {
    // API Gateway builds these before any integration runs, so the caller's
    // origin cannot be reflected the way the Lambdas do it. '*' is safe for
    // these two: they are 401/403 shells carrying no user data, and the app
    // authenticates with an Authorization header rather than cookies, so no
    // credentialed request is ever widened by it.
    'Access-Control-Allow-Origin': "'*'",
    'Access-Control-Allow-Headers': "'Content-Type,Authorization'",
  },
});

const apiResource = api.root.addResource('api');

const apiUser = apiResource.addResource('user');
apiUser.addCorsPreflight({
  allowOrigins: ALLOWED_ORIGINS,
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});

const authStatusRes = apiUser.addResource('auth-status');
authStatusRes.addCorsPreflight({
  allowOrigins: ALLOWED_ORIGINS,
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
  allowOrigins: ALLOWED_ORIGINS,
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
  allowOrigins: ALLOWED_ORIGINS,
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});
setPasswordRes.addMethod('POST', new LambdaIntegration(backend.setPasswordHandler.resources.lambda, {
  proxy: true,
}), {
  authorizer,
  authorizationType: AuthorizationType.COGNITO,
});

const linkGoogleRes = apiUser.addResource('link-google');
linkGoogleRes.addCorsPreflight({
  allowOrigins: ALLOWED_ORIGINS,
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
});
linkGoogleRes.addMethod('POST', new LambdaIntegration(backend.linkGoogleHandler.resources.lambda, {
  proxy: true,
}), {
  authorizer,
  authorizationType: AuthorizationType.COGNITO,
});

// Output API URL for frontend
backend.addOutput({
  custom: {
    authApiUrl: api.url,
    // Google client ID is public (Google ships it in the browser itself), so it
    // can travel through the client config. Sourced from the deploy environment
    // — the app root's .env (loaded by shared/env.ts) or the CI's console
    // variables — so the frontend never needs its own VITE_GOOGLE_CLIENT_ID.
    googleClientId: process.env.GOOGLE_CLIENT_ID ?? '',
  },
});