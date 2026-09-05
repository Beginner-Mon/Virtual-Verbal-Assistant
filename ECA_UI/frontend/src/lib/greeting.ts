/**
 * When the assistant's opening line may be rewritten, and when it may not.
 *
 * The greeting is the one message nobody sent. It is generated on the client
 * from the character's `ui_strings`, never posted to the backend, and rebuilt
 * from scratch when a past session is restored — so it is not transcript, it is
 * chrome that happens to be shaped like a message.
 *
 * That stops being true the moment a real message sits below it. From then on
 * the screen is a record of a conversation, and editing the top of it would
 * claim something was said that never was. This is the same conclusion
 * ChatContext already reached for character switching:
 *
 *     // Subsequent character switches: intentionally no-op — keep original greeting.
 *
 * This function is that rule, extracted so both callers share one copy of it and
 * so it can be tested without a DOM.
 */
export function withGreeting<T extends { id: string; content: string }>(
  messages: T[],
  greetingId: string,
  greeting: string,
): T[] {
  // Empty copy means the character has nothing authored for this locale. The
  // previous language is a worse greeting than the right one, and a far better
  // one than an empty bubble.
  if (!greeting.trim()) return messages

  // Anything below it makes this a transcript rather than an opening screen.
  if (messages.length !== 1) return messages
  if (messages[0].id !== greetingId) return messages

  // Same array when nothing changes: this feeds setMessages, and a fresh array
  // would re-render the whole transcript every time the locale is read.
  if (messages[0].content === greeting) return messages

  return [{ ...messages[0], content: greeting }]
}
