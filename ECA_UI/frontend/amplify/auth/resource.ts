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
          profilePicture: 'picture',
          givenName: 'given_name',
          familyName: 'family_name'
        }
      },
      github: {
        clientId: secret('GITHUB_CLIENT_ID'),
        clientSecret: secret('GITHUB_CLIENT_SECRET'),
        scopes: ['user:email'],
        attributeMapping: {
          email: 'email',
          profilePicture: 'avatar_url',
          givenName: 'name'
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
    profilePicture: { mutable: true }
  },
});
