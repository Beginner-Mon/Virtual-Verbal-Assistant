import { useEffect, useState } from 'react'
import {
  CheckCircle2,
  CreditCard,
  ExternalLink,
  Loader2,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
} from 'lucide-react'
import {
  getBillingConfig,
  getBillingStatus,
  openSandboxPortal,
  startSandboxCheckout,
  type BillingConfig,
  type BillingStatus,
} from '../lib/api'

function billingErrorMessage(error: unknown): string {
  const detail = (
    error as {
      response?: { data?: { detail?: string } }
      message?: string
    }
  )?.response?.data?.detail
  if (detail) return detail
  if (error instanceof Error && error.message) return error.message
  return 'Billing sandbox request failed'
}

export default function BillingContent() {
  const [config, setConfig] = useState<BillingConfig | null>(null)
  const [status, setStatus] = useState<BillingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [action, setAction] = useState<'checkout' | 'portal' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [nextConfig, nextStatus] = await Promise.all([
        getBillingConfig(),
        getBillingStatus(),
      ])
      setConfig(nextConfig)
      setStatus(nextStatus)
    } catch (loadError) {
      setError(billingErrorMessage(loadError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true
    void Promise.all([getBillingConfig(), getBillingStatus()])
      .then(([nextConfig, nextStatus]) => {
        if (!active) return
        setConfig(nextConfig)
        setStatus(nextStatus)
      })
      .catch((loadError: unknown) => {
        if (active) setError(billingErrorMessage(loadError))
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const runAction = async (nextAction: 'checkout' | 'portal') => {
    setAction(nextAction)
    setError(null)
    try {
      if (nextAction === 'checkout') {
        await startSandboxCheckout()
      } else {
        await openSandboxPortal()
      }
    } catch (actionError) {
      setError(billingErrorMessage(actionError))
      setAction(null)
    }
  }

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading billing sandbox…
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-2xl space-y-5">
        <div className="flex items-start gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-500" />
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              Sandbox only — real transactions disabled
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Checkout uses Stripe test data and fake cards. Billing is integrated
              directly with Stripe, and subscription state cannot grant paid
              application access.
            </p>
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/10 p-4">
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-border/50 bg-secondary/20 p-4">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Internal access
            </p>
            <div className="mt-2 flex items-end justify-between gap-3">
              <div>
                <p className="text-2xl font-semibold text-foreground">
                  {status?.access_plan ?? 'DEMO'}
                </p>
                <p className="text-sm text-muted-foreground">Database-controlled plan</p>
              </div>
              <span className="rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-medium text-emerald-500">
                $0 USD
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-border/50 bg-secondary/20 p-4">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Stripe environment
            </p>
            <div className="mt-2 flex items-center gap-2">
              {config?.checkout_enabled ? (
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              ) : (
                <TriangleAlert className="h-5 w-5 text-amber-500" />
              )}
              <p className="text-lg font-semibold text-foreground">
                {config?.checkout_enabled ? 'Sandbox ready' : 'Not configured'}
              </p>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {status?.subscription_status
                ? `Test subscription: ${status.subscription_status}`
                : 'No test subscription recorded'}
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-border/50 p-4">
          <div className="flex items-start gap-3">
            <CreditCard className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-foreground">
                Test subscription checkout
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Stripe will display a simulated subscription amount, but no money moves
                and no real card should be entered.
              </p>
              <div className="mt-3 rounded-lg bg-secondary/40 p-3 text-xs text-muted-foreground">
                Test card: <span className="font-mono text-foreground">4242 4242 4242 4242</span>
                {' · '}Any future expiry{' · '}Any 3-digit CVC
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={!config?.checkout_enabled || action !== null}
                  onClick={() => void runAction('checkout')}
                  className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {action === 'checkout' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ExternalLink className="h-4 w-4" />
                  )}
                  Open sandbox checkout
                </button>
                {status?.has_test_customer && (
                  <button
                    type="button"
                    disabled={!config?.checkout_enabled || action !== null}
                    onClick={() => void runAction('portal')}
                    className="inline-flex items-center gap-2 rounded-lg border border-border/60 px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary/60 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {action === 'portal' ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <CreditCard className="h-4 w-4" />
                    )}
                    Manage test subscription
                  </button>
                )}
                <button
                  type="button"
                  disabled={action !== null}
                  onClick={() => void load()}
                  className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground disabled:opacity-50"
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </div>

        {!config?.checkout_enabled && (
          <p className="text-xs text-muted-foreground">
            To enable testing, configure the three Stripe sandbox values and set{' '}
            <span className="font-mono text-foreground">
              BILLING_SANDBOX_ENABLED=true
            </span>{' '}
            on the backend.
          </p>
        )}
      </div>
    </div>
  )
}
