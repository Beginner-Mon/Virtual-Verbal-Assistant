"""Record a voice sample for VieNeu voice cloning.

Usage:  python record_voice.py
        python record_voice.py --duration 8 --output voices/my_voice.wav
"""
import argparse
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def list_mics():
    devices = sd.query_devices()
    mics = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    return mics


def record(output: str, duration: int, sample_rate: int = 24000):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {duration}s at {sample_rate}Hz...")
    print("Speak now. Press Ctrl+C to cancel.")

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )

        for remaining in range(duration, 0, -1):
            print(f"  {remaining}s...", end="\r")
            sd.sleep(1000)

        sd.wait()
        print(" " * 20, end="\r")

        sf.write(str(output_path), audio, sample_rate)
        size_kb = output_path.stat().st_size / 1024
        print(f"Saved: {output_path} ({size_kb:.0f} KB)")

    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Record voice sample for VieNeu voice cloning")
    parser.add_argument("--duration", type=int, default=8, help="Recording duration in seconds")
    parser.add_argument("--output", type=str, default="voices/my_voice.wav", help="Output file path")
    parser.add_argument("--list", action="store_true", help="List available microphones")
    parser.add_argument("--device", type=int, default=None, help="Microphone device ID")
    args = parser.parse_args()

    if args.list:
        print("Available microphones:")
        for idx, name in list_mics():
            print(f"  [{idx}] {name}")
        return

    if args.device is not None:
        sd.default.device = args.device

    record(output=args.output, duration=args.duration)


if __name__ == "__main__":
    main()
