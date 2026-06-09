import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { CfnUserPoolDomain } from 'aws-cdk-lib/aws-cognito';

const backend = defineBackend({ auth });

// Find the CfnUserPoolDomain that Amplify auto-creates
const cfnDomain = backend.auth.resources.userPool.node
  .tryFindChild('UserPoolDomain')
  ?.node.defaultChild as CfnUserPoolDomain;

if (cfnDomain) {
  cfnDomain.domain = 'eca-us-east-1';
  cfnDomain.addPropertyOverride('ManagedLoginVersion', 2);
}
