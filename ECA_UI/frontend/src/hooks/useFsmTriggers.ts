/**
 * FSM triggers that are tied to the React lifecycle (plan §11.3).
 *
 * Every way the character's state can change, in one place:
 *
 *   | Trigger                  | Where                                     |
 *   |--------------------------|-------------------------------------------|
 *   | Boot (greeting → idle)   | `useFsmBoot` — below                      |
 *   | Timer (idle → bored)     | `useAutoAfterTrigger` — below, declarative |
 *   | One-shot ends            | `AnimationController.handleFinished`       |
 *   | Debug dropdown / file    | `MotionControlPanel` via context          |
 *   | Chat start / answer      | `ChatPanel` SSE callback                  |
 *
 * External-event triggers (the last two) stay explicit at their call site: each
 * arrives from a different source with different payloads, and encoding them as
 * config would mean inventing a condition mini-DSL for a couple of call sites.
 */

import { useEffect, useRef } from 'react'
import type { AnimationController } from '../lib/AnimationController'
import type { AnimationRegistry } from '../lib/AnimationRegistry'
import { STATES, type CharState } from '../lib/AnimationStates'

/**
 * Boot sequence: greet once, then fall through to idle when the greeting clip
 * finishes (plan §2.5).
 *
 * Two non-obvious requirements, both load-bearing:
 *
 *  1. **The idle fallback is mandatory.** `transitionTo` returns false if the
 *     greeting asset is missing, and a controller with no action renders the
 *     bind pose — the exact T-pose the readiness gate exists to prevent.
 *  2. **Greet only once per session.** The controller is recreated whenever the
 *     VRM changes, and React 19 StrictMode double-mounts in dev; without the
 *     module-scoped latch the avatar would wave again on every model swap.
 */
let hasGreeted = false

export function useFsmBoot(
  controller: AnimationController | null,
  registry: AnimationRegistry | null,
): void {
  useEffect(() => {
    if (!controller) return
    let cancelled = false

    const boot = async () => {
      const first: CharState = hasGreeted ? 'idle' : 'greeting'
      hasGreeted = true
      const ok = await controller.transitionTo(first)
      if (cancelled) return
      // Greeting unavailable — must still end up posed.
      if (!ok && first !== 'idle') await controller.transitionTo('idle')
      if (cancelled) return
      // Only once the character is on screen: warm the remaining clips during
      // idle time so the first chat doesn't pay a ~200ms retarget stall.
      registry?.prefetchStatic()
    }

    void boot()
    return () => {
      cancelled = true
    }
  }, [controller, registry])
}

/**
 * Declarative timer trigger: honours `STATES[state].autoAfter`. Replaces the
 * hand-rolled "random idle action" timer that used to live in MotionContext and
 * matched animation labels with a regex.
 */
export function useAutoAfterTrigger(
  controller: AnimationController | null,
  state: CharState,
): void {
  const controllerRef = useRef(controller)
  controllerRef.current = controller

  useEffect(() => {
    const rule = STATES[state].autoAfter
    if (!controller || !rule) return

    const delayMs = (rule.minSec + Math.random() * (rule.maxSec - rule.minSec)) * 1000
    const timer = setTimeout(() => {
      void controllerRef.current?.transitionTo(rule.to)
    }, delayMs)

    return () => clearTimeout(timer)
  }, [controller, state])
}
