import threading

from src.core.state_machine import StateMachine, AssistantState

class Orchestrator:
    def __init__(
        self,
        stt_stage,
        emotion_stage,
        llm_stage,
        tts_stage,
        voice_driver,
        audio_buffer,
        interrupt_controller,
    ):
        self.stt = stt_stage
        self.emotion = emotion_stage
        self.llm = llm_stage
        self.tts = tts_stage
        self.voice_driver = voice_driver
        self.audio_buffer = audio_buffer
        self.interrupt = interrupt_controller

        self.state_machine = StateMachine()

    def handle_text_input(self, user_text: str):
        self.state_machine.set_state(AssistantState.THINKING)

        emotion = self.emotion.process(user_text)
        response = self.llm.process(user_text, emotion)

        self.state_machine.set_state(AssistantState.SPEAKING)

        self.audio_buffer.start()
        self.tts.process(response)

        self.audio_buffer.stop()
        self.state_machine.set_state(AssistantState.IDLE)

        return response

    def handle_voice_input(self, audio_buffer_data, sample_rate: int):
        self.state_machine.set_state(AssistantState.LISTENING)

        user_text = self.stt.process(audio_buffer_data, sample_rate)

        self.state_machine.set_state(AssistantState.THINKING)

        emotion = self.emotion.process(user_text)
        response = self.llm.process(user_text, emotion)

        self.state_machine.set_state(AssistantState.SPEAKING)

        self.tts.process(response)

        self.state_machine.set_state(AssistantState.IDLE)

        return response

    def interrupt_speaking(self):
        if self.state_machine.is_speaking():
            self.interrupt.trigger()
            self.audio_buffer.stop()
            self.state_machine.set_state(AssistantState.IDLE)
            