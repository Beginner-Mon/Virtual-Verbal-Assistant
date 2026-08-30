import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchAuthSession } from 'aws-amplify/auth'

export function useRedirectIfAuthenticated(to = '/') {
  const navigate = useNavigate()
  useEffect(() => {
    fetchAuthSession().then(s => {
      if (s.tokens) navigate(to, { replace: true })
    })
  }, [navigate, to])
}
