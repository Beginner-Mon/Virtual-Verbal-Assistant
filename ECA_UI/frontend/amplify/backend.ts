import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { CfnUserPool, CfnUserPoolDomain } from 'aws-cdk-lib/aws-cognito';

const backend = defineBackend({ auth });

// Find the CfnUserPoolDomain that Amplify auto-creates
const cfnDomain = backend.auth.resources.userPool.node
  .tryFindChild('UserPoolDomain')
  ?.node.defaultChild as CfnUserPoolDomain;

if (cfnDomain) {
  cfnDomain.domain = 'eca-us-east-1';
  cfnDomain.addPropertyOverride('ManagedLoginVersion', 2);
}

// Patch missing AttributeDataType on UserPool schema entries (Cognito now requires it)
const cfnUserPool = backend.auth.resources.userPool.node.defaultChild as CfnUserPool;
if (cfnUserPool) {
  // email, given_name, family_name, picture — all String type
  for (let i = 0; i <= 3; i++) {
    cfnUserPool.addPropertyOverride(`Schema.${i}.AttributeDataType`, 'String');
  }
}
