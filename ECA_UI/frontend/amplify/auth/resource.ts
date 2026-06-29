import { defineAuth, secret } from '@aws-amplify/backend';

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
      callbackUrls: [
        'http://localhost:5173/',
      ],
      logoutUrls: [
        'http://localhost:5173/',
      ],
    }
  },
  userAttributes: {
    preferredUsername: { required: false, mutable: true },
    givenName: { required: false, mutable: true },
    familyName: { required: false, mutable: true },
  },
});
