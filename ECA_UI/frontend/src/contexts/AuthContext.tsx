import { createContext, useContext } from 'react'

export interface AuthUser {
  signInDetails?: { loginId?: string }
}

export interface FetchUserAttributesOutput {
  sub?: string
  email?: string
  email_verified?: string
  given_name?: string
  family_name?: string
  phone_number?: string
  [key: string]: string | undefined
}

export interface UserContextType {
  signOut?: () => void
  user?: AuthUser
  userAttributes?: FetchUserAttributesOutput
}

export const AuthContext = createContext<UserContextType>({})

export function useAuth() {
  return useContext(AuthContext)
}
