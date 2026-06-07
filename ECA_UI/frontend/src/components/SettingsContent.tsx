import { useState } from 'react'
import { KeyRound, Bot, Sparkles, Brain, Cpu, Zap, Wind, Server } from 'lucide-react'

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
              onClick={onNavigateToProviders}
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

function ProviderDetailView({ provider }: { provider?: { id: string; name: string } }) {
  const [apiKey, setApiKey] = useState('')
  const [status, setStatus] = useState<'idle' | 'checking' | 'connected' | 'failed'>('idle')

  const checkStatus = async () => {
    setStatus('checking')
    // Placeholder for actual connection check
    await new Promise((r) => setTimeout(r, 1500))
    setStatus(apiKey.length > 0 ? 'connected' : 'failed')
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-foreground mb-1">Insert API Key</h1>
        <p className="text-sm text-muted-foreground">Enter your {provider?.name ?? ''} API key to connect</p>
      </div>

      <input
        type="password"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        placeholder="Paste your API key here..."
        className="w-full px-4 py-3 rounded-xl border border-border/40 bg-card text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-foreground transition-colors"
      />

      <div className="border-t border-border/40 pt-6">
        <h2 className="text-base font-semibold text-foreground mb-3">Status</h2>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                status === 'connected'
                  ? 'bg-green-500'
                  : status === 'failed'
                    ? 'bg-red-500'
                    : status === 'checking'
                      ? 'bg-yellow-500 animate-pulse'
                      : 'bg-muted-foreground/40'
              }`}
            />
            <span className="text-sm text-muted-foreground">
              {status === 'connected'
                ? 'Connected'
                : status === 'failed'
                  ? 'Connection failed'
                  : status === 'checking'
                    ? 'Checking...'
                    : 'Not connected'}
            </span>
          </div>
          <button
            onClick={checkStatus}
            disabled={status === 'checking'}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            Check Connection
          </button>
        </div>
      </div>
    </div>
  )
}