import numpy as np
import sounddevice as sd
import librosa


# 🎤 UNIVERSAL RECORD FUNCTION
def record_audio(duration=3, fs=44100):
    print(f"🎤 Recording for {duration} seconds...")

    try:
        sd.default.channels = 1  # ✅ force mono (safe)
        devices = sd.query_devices()

        input_devices = []

        # ✅ find all input devices
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(i)

        if not input_devices:
            print("❌ No microphone devices found!")
            return None, None

        # ✅ try each device
        for device_id in input_devices:
            try:
                print(f"🔍 Trying device {device_id}...")

                recording = sd.rec(
                    int(duration * fs),
                    samplerate=fs,
                    channels=1,
                    dtype='float32',
                    device=device_id
                )

                sd.wait()
                print(f"✅ Recording success with device {device_id}")

                return recording.flatten(), fs

            except Exception as ex:
                print(f"❌ Device {device_id} failed:", ex)

        print("❌ No working microphone found!")
        return None, None

    except Exception as ex:
        print("❌ Microphone Error:", ex)
        return None, None


# 🎧 ANALYSIS FUNCTION
def analyze_voice(audio, fs):
    if audio is None:
        raise ValueError("No audio data")

    # 🎯 Pitch
    try:
        pitch_values = librosa.yin(audio, fmin=50, fmax=300)
        pitch = np.mean(pitch_values)
    except:
        pitch = 0

    # 🔊 Energy
    energy = np.mean(audio**2)

    # 🥁 Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=fs)
        if isinstance(tempo, np.ndarray):
            tempo = float(np.mean(tempo))
        else:
            tempo = float(tempo)
    except:
        tempo = 0.0

    return pitch, energy, tempo


# 📊 SCORE CALCULATION
def calculate_score(pitch, energy, tempo):
    score = 0

    if pitch > 185:  # Realistic high pitch threshold
        score += 1

    if energy > 0.02:  # Moderate loudness check
        score += 1

    if 0 < tempo < 75:  # Slightly slow speaking
        score += 1

    return score


# 🧠 RESULT CLASSIFICATION
def classify_result(score):
    if score >= 2: # Kam se kam 2 indicators milne par hi 'Lie' declare karein
        return "POSSIBLE LIE"
    else:
        return "LIKELY TRUTH"


# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    audio, fs = record_audio(duration=5)

    if audio is not None:
        pitch, energy, tempo = analyze_voice(audio, fs)

        score = calculate_score(pitch, energy, tempo)

        result = classify_result(score)

        print("\n📊 RESULTS:")
        print(f"Average Pitch: {pitch:.2f} Hz")
        print(f"Average Energy: {energy:.4f}")
        print(f"Speech Tempo: {tempo:.2f} BPM")
        print(f"Score: {score}")
        print(f"Result: {result}")
    else:
        print("❌ Recording failed. Try again.")