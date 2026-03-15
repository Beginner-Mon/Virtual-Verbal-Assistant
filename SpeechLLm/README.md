
Virtual Verbal Assistant (SpeechLLM)

Overview
This project implements a modular streaming voice assistant capable of processing speech input,
generating intelligent responses using a language model, synthesizing speech output, and optionally
triggering motion or animation responses. The system is designed with a pipeline architecture that
supports low-latency streaming interaction.

Architecture
User Speech
    ↓
Speech-to-Text (Whisper)
    ↓
Emotion Detection
    ↓
LLM Response Generation (Phi-3)
    ↓
Text-to-Speech (Coqui / ElevenLabs)
    ↓
Motion Controller
    ↓
Audio + Motion Output

Project Structure
configs/
    base.yaml
    models.yaml

data/

models/

src/
    context/
        memory_manager.py
        prompt_template.py

    core/
        events.py
        orchestrator.py
        state_machine.py

    services/
        coqui_client.py
        elevenlabs_client.py
        phi3_client.py
        tts_router.py
        voice_driver.py
        whisper_client.py

    stages/
        emotion_stage.py
        llm_stage.py
        stt_stage.py
        tts_stage.py
        motion_stage.py

    utils/
        logger.py
        action_normalizer.py

streaming/
    audio_stream_buffer.py
    interrupt_controller.py
    token_streamer.py

main.py
requirements.txt
.gitignore

Core Modules

Context Management
memory_manager.py
Maintains conversation history and contextual memory for the assistant.

prompt_template.py
Constructs structured prompts used by the language model.

Core Engine

orchestrator.py
Coordinates the execution of all pipeline stages.

state_machine.py
Controls the state transitions of the assistant.

events.py
Defines system-level events used for communication between modules.

Services

whisper_client.py
Handles speech recognition using Whisper.

phi3_client.py
Interface for the Phi-3 language model.

coqui_client.py
Local text-to-speech generation using Coqui.

elevenlabs_client.py
Cloud-based text-to-speech generation using ElevenLabs.

tts_router.py
Selects which TTS engine to use.

voice_driver.py
Manages audio playback.

Processing Stages

stt_stage.py
Processes incoming speech and converts it to text.

emotion_stage.py
Detects emotional context from input text.

llm_stage.py
Generates responses using the language model.

tts_stage.py
Converts text responses into speech.

motion_stage.py
Triggers motion or animation events.

Streaming System

token_streamer.py
Streams tokens from the language model for real-time responses.

audio_stream_buffer.py
Buffers audio chunks for smooth playback.

interrupt_controller.py
Allows users to interrupt the assistant mid-response.

Requirements

Install dependencies:
pip install -r requirements.txt

Major libraries used include:
PyTorch
Faster-Whisper
Transformers
Coqui TTS
ElevenLabs API
Ollama

Model Setup

Install Ollama and download the Phi-3 model:
ollama pull phi3:3.8b

Whisper models will automatically download during first execution.

Running the Assistant

Run the main entry point:
python main.py

The assistant will:
1. Capture microphone input
2. Convert speech to text
3. Generate an LLM response
4. Convert the response to speech
5. Trigger motion events

Features

Streaming responses
Interruptible conversation
Multiple TTS backends
Modular pipeline architecture
Emotion-aware responses
Motion integration

Configuration

configs/base.yaml
configs/models.yaml

These configuration files control model selection, streaming behavior,
voice parameters, and system settings.

Future Improvements

Multilingual support
Emotion-conditioned TTS
Avatar motion synthesis
GPU acceleration
Long-term memory integration

