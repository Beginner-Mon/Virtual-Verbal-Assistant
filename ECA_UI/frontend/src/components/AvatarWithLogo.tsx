import { Avatar, AvatarFallback } from './ui/avatar'
import EcaLogo from './EcaLogo'
import { cn } from '@/lib/utils'

const SIZES = {
  xs: { avatar: 'w-7 h-7', logo: 'w-6 h-6' },
  sm: { avatar: 'w-9 h-9', logo: 'w-9 h-9' },
  md: { avatar: 'w-14 h-14', logo: 'w-14 h-14' },
  lg: { avatar: 'w-20 h-20', logo: 'w-20 h-20' },
} as const

interface AvatarWithLogoProps {
  size: keyof typeof SIZES
  bgClassName?: string
  logoClassName?: string
}

export default function AvatarWithLogo({ size, bgClassName, logoClassName }: AvatarWithLogoProps) {
  const s = SIZES[size]
  return (
    <Avatar className={s.avatar}>
      <AvatarFallback className={cn(bgClassName ?? 'bg-muted-foreground/20', logoClassName)}>
        <EcaLogo className={s.logo} />
      </AvatarFallback>
    </Avatar>
  )
}
