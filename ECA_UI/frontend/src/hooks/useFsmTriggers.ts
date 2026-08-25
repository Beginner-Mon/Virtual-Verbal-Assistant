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
 * Boot sequence: ensure the model has a pose (idle) immediately after load.
 *
 * Greeting is now handled per-VRM in MotionContext (Plan B1 visitedRef) — boot
 * no longer greets globally. The previous global `hasGreeted` latch made the
 * first VRM greet via boot while subsequent VRMs greeted via MotionContext,
 * splitting the source. Now boot's only job is to guarantee a pose and warm
 * clips; per-VRM greeting is owned by MotionContext's visited guard.
 *
 * The idle fallback is still mandatory: if idle fails, model stays in bind pose.
 */
export function useFsmBoot(
  controller: AnimationController | null,
  registry: AnimationRegistry | null,
): void {
  useEffect(() => {
    if (!controller) return
    let cancelled = false

    const boot = async () => {
      const ok = await controller.transitionTo('idle')
      if (cancelled) return
      if (!ok) console.warn('[useFsmBoot] idle transition failed — model may be in bind pose')
      if (cancelled) return
      // Warm remaining clips during idle time so first chat doesn't pay retarget stall.
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
