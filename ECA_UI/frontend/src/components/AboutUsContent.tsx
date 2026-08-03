import { Info } from 'lucide-react'

export default function AboutUsContent() {
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
              <span className="text-sm font-semibold text-foreground">Project ECA</span>
              <span className="text-xs text-muted-foreground ml-2">by Ordinary Studio</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              An interactive multimodal AI assistant combining conversational AI, 3D motion synthesis,
              and voice I/O. Built for those passionate about real-time 3D character animation and
              AI-driven motion generation — no pre-recorded animations required.
            </p>
          </div>
        </section>

        <hr className="border-border/40" />

        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            Who We Are
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">Ordinary Studio</span> — a team of
            final-year students who started this project as a graduation assignment. We are
            driven by a shared passion for real-time 3D rendering, AI, and human-computer
            interaction, aiming to push the boundaries of what virtual assistants can be.
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
