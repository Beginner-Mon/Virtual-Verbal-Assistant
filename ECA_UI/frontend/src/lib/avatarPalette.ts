export const AVATAR_BG_OPTIONS = [
  { id: 'slate', className: 'bg-muted-foreground/20', logoClassName: 'text-muted-foreground', value: '#e5e7eb' },
  { id: 'violet', className: 'bg-violet-500', logoClassName: 'text-violet-100', value: '#8b5cf6' },
  { id: 'blue', className: 'bg-sky-500', logoClassName: 'text-sky-100', value: '#0ea5e9' },
  { id: 'emerald', className: 'bg-emerald-500', logoClassName: 'text-emerald-100', value: '#10b981' },
  { id: 'amber', className: 'bg-amber-500', logoClassName: 'text-amber-950', value: '#f59e0b' },
  { id: 'rose', className: 'bg-rose-500', logoClassName: 'text-rose-100', value: '#f43f5e' },
  { id: 'cyan', className: 'bg-cyan-500', logoClassName: 'text-cyan-50', value: '#06b6d4' },
  { id: 'indigo', className: 'bg-indigo-500', logoClassName: 'text-indigo-100', value: '#6366f1' },
] as const

export type AvatarBgId = (typeof AVATAR_BG_OPTIONS)[number]['id']
