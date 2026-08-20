import * as React from 'react'
import { Eye, EyeOff } from 'lucide-react'

import { cn } from '@/lib/utils'

/**
 * Password field with a reveal toggle.
 *
 * The six password inputs across the auth pages all repeated the same class
 * string verbatim, so the styling lives here rather than being restated at each
 * call site. `type` is owned by this component — everything else passes through,
 * which is why swapping a call site over is a rename and nothing more.
 */
function PasswordInput({
  className,
  ...props
}: Omit<React.ComponentProps<'input'>, 'type'>) {
  const [visible, setVisible] = React.useState(false)

  return (
    <div className="relative">
      <input
        type={visible ? 'text' : 'password'}
        className={cn(
          'w-full text-sm text-foreground bg-secondary/40 rounded-lg pl-3 pr-10 py-2.5',
          'border border-border/30 outline-none focus:border-primary/50 transition-colors',
          // Edge ships its own reveal button inside password fields; without this
          // there are two eyes sitting next to each other.
          '[&::-ms-reveal]:hidden [&::-ms-clear]:hidden',
          className,
        )}
        {...props}
      />
      <button
        type="button"
        onClick={() => setVisible((v) => !v)}
        // Keeps the caret in the field. Without it the click moves focus to the
        // button and typing needs another click on the input first.
        onMouseDown={(e) => e.preventDefault()}
        aria-label={visible ? 'Hide password' : 'Show password'}
        aria-pressed={visible}
        className="
          absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md
          text-muted-foreground hover:text-foreground hover:bg-secondary/60
          transition-colors cursor-pointer
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
        "
      >
        {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
      </button>
    </div>
  )
}

export { PasswordInput }
