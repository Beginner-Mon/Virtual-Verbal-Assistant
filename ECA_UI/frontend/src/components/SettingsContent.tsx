import { KeyRound } from 'lucide-react'

export default function SettingsContent() {
  const items = [
    { id: 'providers', icon: KeyRound, label: 'Select Providers', description: 'Configure your LLM API providers and keys' },
  ]

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="space-y-4">
        {items.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              className="flex justify-between items-stretch w-full rounded-xl border border-border/40 bg-card hover:border-foreground transition-colors text-left group overflow-hidden"
            >
              <div className="flex flex-col justify-center py-5 px-6">
                <span className="text-base font-semibold text-foreground">{item.label}</span>
                <p className="text-sm text-muted-foreground mt-1">{item.description}</p>
              </div>
              <div className="flex items-end justify-end w-20 shrink-0 pr-1">
                <Icon className="w-[72px] h-[72px] text-muted-foreground group-hover:scale-125 group-hover:text-foreground transition-all duration-300" />
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
