import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar'
import EcaLogo from './EcaLogo'

const SIZES = {
  xs: { avatar: 'w-7 h-7', logo: 'w-6 h-6' },
  sm: { avatar: 'w-9 h-9', logo: 'w-9 h-9' },
  md: { avatar: 'w-14 h-14', logo: 'w-14 h-14' },
  lg: { avatar: 'w-20 h-20', logo: 'w-20 h-20' },
} as const

interface AvatarWithLogoProps {
  size: keyof typeof SIZES
  profilePicture?: string
}

export default function AvatarWithLogo({ size, profilePicture }: AvatarWithLogoProps) {
  const s = SIZES[size]
  return (
    <Avatar className={s.avatar}>
      <AvatarImage src={profilePicture} alt="Avatar" referrerPolicy="no-referrer" />
      <AvatarFallback className="bg-muted-foreground/20" style={{ color: 'var(--muted-foreground)' }}>
        <EcaLogo className={s.logo} />
      </AvatarFallback>
    </Avatar>
  )
}
