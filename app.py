import numpy as np
import time  # Added for session timing
from flask import Flask, Response, render_template, jsonify
from face_test import generate_frames, shared_data
from voice import record_audio, analyze_voice, calculate_lie_probability, classify_voice_result
from result import calculate_final_verdict

app = Flask(__name__)

# --- GLOBAL STORAGE ---
voice_baseline = None 
session_history = {
    "is_active": False,
    "tension_count": 0,
    "total_frames": 0,
    "voice_scores": [],
    "start_blink": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    # Pass session_history to the generator if needed, 
    # or just let generate_frames update shared_data
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate_voice')
def calibrate_voice():
    global voice_baseline
    try:
        audio, fs = record_audio(duration=3)
        if audio is not None:
            voice_baseline = analyze_voice(audio, fs)
            return jsonify({'status': 'Voice Calibrated', 'baseline': voice_baseline[0]})
        return jsonify({'error': 'Mic failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- NEW: START 1-MINUTE SESSION ---
@app.route('/start_session')
def start_session():
    global session_history
    session_history["is_active"] = True
    session_history["tension_count"] = 0
    session_history["total_frames"] = 0
    session_history["voice_scores"] = []
    session_history["start_blink"] = shared_data.get("blink_count", 0)
    return jsonify({"status": "Session Started", "duration": 60})

# --- UPDATED: VOICE ANALYSIS (Now records to history) ---
@app.route('/voice_analysis')
def voice_analysis():
    global voice_baseline, session_history
    try:
        if voice_baseline is None:
            return jsonify({'error': 'Calibrate first'}), 400

        audio, fs = record_audio(duration=3)
        if audio is None: return jsonify({'error': 'Mic error'}), 500

        current_stats = analyze_voice(audio, fs)
        voice_prob = calculate_lie_probability(current_stats, voice_baseline)
        
        # Track for final summary
        if session_history["is_active"]:
            session_history["voice_scores"].append(voice_prob)
            # Track tension from shared_data during this voice clip
            session_history["total_frames"] += 1
            if shared_data.get("facial_prediction") == "Tense":
                session_history["tension_count"] += 1

        return jsonify({
            'voice_probability': voice_prob,
            'result': classify_voice_result(voice_prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- NEW: GET FINAL 1-MINUTE SUMMARY ---
# --- NEW: GET FINAL 1-MINUTE SUMMARY (33.33% Model) ---
@app.route('/get_session_report')
def get_session_report():
    global session_history
    session_history["is_active"] = False # Stop tracking
    
    # 1. Gather raw data from the session
    total_blinks = shared_data.get("blink_count", 0) - session_history["start_blink"]
    
    # Calculate how consistent the facial tension was
    tension_consistency = (session_history["tension_count"] / session_history["total_frames"] * 100) if session_history["total_frames"] > 0 else 0
    
    # Average the voice stress probabilities recorded during the minute
    avg_voice_prob = sum(session_history["voice_scores"]) / len(session_history["voice_scores"]) if session_history["voice_scores"] else 0

    # 2. Call the 33.33% weighted model from Result.py
    final_prob, verdict = calculate_final_verdict(total_blinks, avg_voice_prob, tension_consistency)

    # 3. Return the detailed report to the UI
    return jsonify({
        "verdict": verdict,
        "lie_probability": f"{final_prob}%",
        "blink_rate": f"{total_blinks} blinks/min",
        "voice_stress": f"{round(avg_voice_prob, 1)}%",
        "face_tension": f"{round(tension_consistency, 1)}%"
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)