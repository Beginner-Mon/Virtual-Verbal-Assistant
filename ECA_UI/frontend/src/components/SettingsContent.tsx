import { KeyRound, Bot, Sparkles, Brain, Cpu, Zap, Wind, Server, MonitorCog, Info, ScrollText } from 'lucide-react'
import ProviderDetailView from './ProviderDetailView'

interface SettingsContentProps {
  view?: 'main' | 'providers' | 'provider-detail'
  onNavigateToProviders?: () => void
  onSelectProvider?: (provider: { id: string; name: string }) => void
  selectedProvider?: { id: string; name: string }
}

export default function SettingsContent({ view = 'main', onNavigateToProviders, onSelectProvider, selectedProvider }: SettingsContentProps) {
  if (view === 'provider-detail') {
    return <ProviderDetailView provider={selectedProvider} />
  }

  if (view === 'providers') {
    const providers = [
      { id: 'vva', name: 'VVA', icon: Server, official: true },
      { id: 'anthropic', name: 'Anthropic', icon: Bot },
      { id: 'gemini', name: 'Gemini', icon: Sparkles },
      { id: 'deepseek', name: 'DeepSeek', icon: Brain },
      { id: 'openai', name: 'OpenAI', icon: Cpu },
      { id: 'groq', name: 'Groq', icon: Zap },
      { id: 'mistral', name: 'Mistral', icon: Wind },
    ]

    return (
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {providers.map((provider) => {
            const Icon = provider.icon
            return (
              <button
                key={provider.id}
                onClick={() => {
                  if (!provider.official) {
                    onSelectProvider?.({ id: provider.id, name: provider.name })
                  }
                }}
                className="flex justify-between items-center w-full rounded-xl border border-border/40 bg-card hover:border-foreground transition-colors group overflow-hidden"
              >
                <div className="flex flex-col justify-center py-5 px-6">
                  <div className="flex items-center gap-2">
                    <span className="text-base font-semibold text-foreground">{provider.name}</span>
                    {provider.official && (
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-primary bg-primary/10 px-2 py-0.5 rounded-full">Official</span>
                    )}
                  </div>
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

  const items = [
    { id: 'providers', icon: KeyRound, label: 'Select Providers', description: 'Configure your LLM API providers and keys', onClick: onNavigateToProviders },
    { id: 'graphics', icon: MonitorCog, label: 'Graphic Settings', description: 'Adjust visual quality and performance options' },
    { id: 'about', icon: Info, label: 'About Us', description: 'Learn more about our team and mission' },
    { id: 'terms', icon: ScrollText, label: 'Terms of Service', description: 'Read our terms and conditions' },
  ]

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="space-y-4">
        {items.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={item.onClick}
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