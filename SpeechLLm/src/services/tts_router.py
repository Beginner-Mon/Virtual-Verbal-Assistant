class TTSRouter:
	"""Routes TTS requests to ElevenLabs with Coqui fallback."""

	def __init__(self, eleven_client=None, coqui_client=None):
		self.eleven = eleven_client
		self.coqui = coqui_client
		self.last_provider = None

	def synthesize(self, text: str, language: str = "en") -> str:
		# Prefer ElevenLabs in normal operation.
		if self.eleven:
			try:
				audio_path = self.eleven.synthesize(text)
				self.last_provider = "elevenlabs"
				return audio_path
			except Exception as e:
				error_msg = str(e).lower()
				# Fall back to Coqui when quota/credits are exhausted.
				if "quota_exceeded" in error_msg or "credits" in error_msg:
					print("[TTS Router] ElevenLabs credits exhausted. Falling back to Coqui.")
				else:
					print(f"[TTS Router] ElevenLabs failed, trying Coqui: {e}")

		if self.coqui:
			audio_path = self.coqui.synthesize(text, language=language)
			self.last_provider = "coqui"
			return audio_path

		raise RuntimeError("All TTS providers failed (ElevenLabs and Coqui unavailable).")