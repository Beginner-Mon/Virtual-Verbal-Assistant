import { Settings, LogOut, Sun, Moon, User } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'
import { useTheme } from '../../contexts/ThemeContext'

export default function SettingsPanel() {
  const { signOut, user } = useAuth()
  const { theme, setTheme } = useTheme()

  const email = user?.signInDetails?.loginId ?? 'Unknown user'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <Settings className="w-4 h-4 text-primary" />
          Settings
        </h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {/* User info */}
        <div className="flex items-center gap-3 p-3 rounded-xl bg-secondary/40 border border-border/30">
          <div className="w-10 h-10 rounded-full bg-primary/15 flex items-center justify-center shrink-0">
            <User className="w-5 h-5 text-primary" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs text-muted-foreground">Signed in as</p>
            <p className="text-sm font-medium text-foreground truncate">{email}</p>
          </div>
        </div>

        {/* Theme toggle */}
        <div className="flex items-center justify-between p-3 rounded-xl bg-secondary/40 border border-border/30">
          <div className="flex items-center gap-3">
            {theme === 'dark' ? (
              <Moon className="w-4 h-4 text-muted-foreground" />
            ) : (
              <Sun className="w-4 h-4 text-muted-foreground" />
            )}
            <span className="text-sm text-foreground">
              {theme === 'dark' ? 'Dark' : 'Light'} mode
            </span>
          </div>
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className={`
              relative w-11 h-6 rounded-full transition-colors duration-200
              ${theme === 'dark' ? 'bg-primary' : 'bg-border'}
            `}
          >
            <span
              className={`
                absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm
                transition-transform duration-200
                ${theme === 'dark' ? 'translate-x-5' : 'translate-x-0'}
              `}
            />
          </button>
        </div>
      </div>

      {/* Logout at bottom */}
      <div className="p-3 border-t border-border/40 shrink-0">
        <button
          onClick={() => signOut?.()}
          className="flex items-center gap-3 w-full p-3 rounded-xl text-sm font-medium text-destructive hover:bg-destructive/10 transition-colors"
        >
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
    </div>
  )
}
