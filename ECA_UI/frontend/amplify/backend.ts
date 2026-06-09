import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { CfnUserPoolDomain, CfnUserPool } from 'aws-cdk-lib/aws-cognito';

const backend = defineBackend({ auth });

// Disable CAPTCHA ("I'm not a robot") — turn off advanced security
const cfnUserPool = backend.auth.resources.userPool.node.defaultChild as CfnUserPool;
cfnUserPool.userPoolAddOns = { advancedSecurityMode: 'OFF' };

// Find the CfnUserPoolDomain that Amplify auto-creates
const cfnDomain = backend.auth.resources.userPool.node
  .tryFindChild('UserPoolDomain')
  ?.node.defaultChild as CfnUserPoolDomain;

if (cfnDomain) {
  cfnDomain.domain = 'eca-us-east-1';
  cfnDomain.addPropertyOverride('ManagedLoginVersion', 2);
}
