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
def calculate_lie_probability(current_stats, baseline_stats):
    """
    Compares current voice to baseline to find 'Stress'
    """
    c_pitch, c_energy, c_tempo = current_stats
    b_pitch, b_energy, b_tempo = baseline_stats
    
    score = 0
    
    # 1. Pitch Spike (Strongest Indicator)
    # If pitch increases by more than 10% from natural baseline
    if c_pitch > (b_pitch * 1.10):
        score += 50 
        
    # 2. Energy/Volume Increase (Aggression/Defensiveness)
    if c_energy > (b_energy * 1.5):
        score += 25
        
    # 3. Tempo Change (Hesitation or Rushing)
    # If they speak 20% faster or slower than their normal speed
    if c_tempo > (b_tempo * 1.2) or c_tempo < (b_tempo * 0.8):
        score += 25

    return score # Returns a probability out of 100

def classify_voice_result(probability):
    if probability >= 75:
        return "HIGH STRESS (Potential Lie)"
    elif probability >= 40:
        return "MODERATE STRESS"
    else:
        return "STABLE (Likely Truth)"
# 🚀 TEST EXECUTION (Put this at the very bottom of voice.py)
if __name__ == "__main__":
    print("--- STEP 1: CALIBRATION (Speak Normally) ---")
    audio_b, fs_b = record_audio(duration=3)
    if audio_b is not None:
        baseline_stats = analyze_voice(audio_b, fs_b)
        print(f"Baseline Set: Pitch={baseline_stats[0]:.2f}")

        print("\n--- STEP 2: LIE TEST (Speak with Stress/High Pitch) ---")
        audio_t, fs_t = record_audio(duration=3)
        current_stats = analyze_voice(audio_t, fs_t)

        prob = calculate_lie_probability(current_stats, baseline_stats)
        result = classify_voice_result(prob)

        print(f"\n📊 RESULT: {result} ({prob}%)")
    else:
        print("❌ Recording failed.")