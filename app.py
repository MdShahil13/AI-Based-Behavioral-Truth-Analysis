import numpy as np
from flask import Flask, Response, render_template, jsonify
from face_test import generate_frames, shared_data
from voice import record_audio, analyze_voice, calculate_score, classify_result

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/blink_count')
def blink_count():
    return jsonify(count=shared_data["blink_count"])

@app.route('/voice_analysis')
def voice_analysis():
    try:
        # 1. Recording se pehle ke blinks
        start_blinks = shared_data.get("blink_count", 0)
        
        audio, fs = record_audio()

        if audio is None:
            return jsonify({'error': 'Microphone not working'}), 500

        # 2. Recording ke baad ke blinks
        end_blinks = shared_data.get("blink_count", 0)
        blinks_during_speech = end_blinks - start_blinks

        # 3. Voice Analysis
        pitch, energy, tempo = analyze_voice(audio, fs)
        voice_score = calculate_score(pitch, energy, tempo)
        
        # 4. Facial Result (Stream se current status)
        face_result = shared_data.get("facial_prediction", "Natural")

        # 5. Combined Logic (Final Verdict)
        final_score = voice_score
        if face_result == "Tense":
            final_score += 2
        if blinks_during_speech > 5: # Agar 3 sec mein 5+ baar blink kiya toh stress
            final_score += 1
            
        final_result = "POSSIBLE LIE" if final_score >= 2 else "LIKELY TRUTH"

        if isinstance(tempo, (np.ndarray, list, tuple)):
            tempo = float(np.mean(tempo))
        else:
            tempo = float(tempo)

        return jsonify({
            'pitch': float(round(pitch, 2)),
            'energy': float(round(energy, 4)),
            'tempo': float(round(tempo, 2)),
            'voice_score': float(voice_score),
            'face_prediction': face_result,
            'blinks_during_speech': blinks_during_speech,
            'result': final_result,
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    
    app.run(debug=True, use_reloader=False, threaded=True)
