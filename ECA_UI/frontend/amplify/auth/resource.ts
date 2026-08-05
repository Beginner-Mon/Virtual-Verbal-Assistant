import { defineAuth, secret } from '@aws-amplify/backend';
import { OAUTH_REDIRECT_URLS } from '../shared/origins';

export const auth = defineAuth({
  loginWith: {
    email: true,
    externalProviders: {
      google: {
        clientId: secret('GOOGLE_CLIENT_ID'),
        clientSecret: secret('GOOGLE_CLIENT_SECRET'),
        scopes: ['email', 'profile', 'openid'],
        attributeMapping: {
          email: 'email',
          emailVerified: 'email_verified',
          givenName: 'given_name',
          familyName: 'family_name',
        }
      },
      // Includes the `ecaapp://callback/` custom scheme so the system browser
      // can hand control back to an installed Capacitor build. Cognito accepts
      // custom schemes; the same URI must also be registered in the Google
      // console and declared by the native project.
      callbackUrls: OAUTH_REDIRECT_URLS,
      logoutUrls: OAUTH_REDIRECT_URLS,
    }
  },
  userAttributes: {
    preferredUsername: { required: false, mutable: true },
    givenName: { required: false, mutable: true },
    familyName: { required: false, mutable: true },
  },
});
