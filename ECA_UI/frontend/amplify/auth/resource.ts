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
    givenName: { required: true, mutable: true },
    familyName: { required: true, mutable: true },
  },
});
