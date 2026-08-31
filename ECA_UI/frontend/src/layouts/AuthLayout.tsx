import type { ReactNode } from 'react'

/**
 * Shared shell for the four auth screens (/login, /enter-password,
 * /create-account, /set-password).
 *
 * All four had spelled out the same centred wrapper, so the studio mark had
 * nowhere to live except four copies of it. Takes `children` rather than an
 * <Outlet> because the pages are separate routes — /set-password sits behind
 * AuthGuard while the rest are public — so there is no single route to nest.
 */
export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-6 space-y-6">{children}</div>
    </div>
  )
}
