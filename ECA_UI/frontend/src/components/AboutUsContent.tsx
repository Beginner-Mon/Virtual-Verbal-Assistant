import { Info } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export default function AboutUsContent() {
  const { t } = useTranslation()
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="flex items-center gap-2 mb-6">
        <Info className="w-5 h-5 text-muted-foreground" />
        <h2 className="text-lg font-semibold text-foreground">About Us</h2>
      </div>

      <div className="space-y-6">
        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            Introduction
          </h3>
          <div className="space-y-3">
            <div>
              <span className="text-sm font-semibold text-foreground">{t('about.product')}</span>
              <span className="text-xs text-muted-foreground ml-2">{t('about.by')}</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {t('about.summary')}
            </p>
          </div>
        </section>

        <hr className="border-border/40" />

        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            {t('about.who_we_are')}
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">{t('about.studio')}</span>{' '}
            {t('about.who_we_are_body')}
          </p>
        </section>

        <hr className="border-border/40" />

        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            Contact
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Have questions or want to collaborate? Reach out to us at{' '}
            <a
              href="mailto:contact@ordinary.studio"
              className="text-primary hover:underline font-medium"
            >
              contact@ordinary.studio
            </a>
          </p>
        </section>
      </div>
    </div>
  )
}
