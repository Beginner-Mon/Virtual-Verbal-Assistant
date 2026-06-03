import { createContext, useContext } from 'react'

export interface UserContextType {
  signOut?: () => void
  user?: { signInDetails?: { loginId?: string } }
}

export const AuthContext = createContext<UserContextType>({})

export function useAuth() {
  return useContext(AuthContext)
}
