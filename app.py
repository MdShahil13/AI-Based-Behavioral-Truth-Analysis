import numpy as np
import time
from functools import wraps
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session, g
from face_test import generate_frames, shared_data
from voice import record_audio, analyze_voice, calculate_lie_probability, classify_voice_result
from result import calculate_final_verdict

# ✅ Import Blueprint
from auth import auth

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ✅ Register Blueprint
app.register_blueprint(auth)

# --- GLOBAL STORAGE ---
voice_baseline = None 

# Session tracking for the 1-minute analysis
session_history = {
    "is_active": False,
    "tension_count": 0,
    "total_frames": 0,
    "voice_scores": [],
    "start_blink": 0
}

# ---------------- LOGIN REQUIRED ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function


# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('home.html')



from db import users_collection

@app.route('/app')
@login_required
def app_main():
    # Find username from session['user'] (which is email)
    user_doc = users_collection.find_one({"email": session.get('user')})
    g.username = user_doc['username'] if user_doc else 'User'
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- CALIBRATE VOICE ----------------
@app.route('/calibrate_voice')
def calibrate_voice():
    global voice_baseline
    try:
        audio, fs = record_audio(duration=3)
        if audio is not None:
            voice_baseline = analyze_voice(audio, fs)
            pitch = voice_baseline[0] if isinstance(voice_baseline, tuple) else voice_baseline
            return jsonify({'status': 'Voice Calibrated', 'baseline': pitch})
        return jsonify({'error': 'Mic failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- SESSION START ----------------
@app.route('/start_session')
def start_session():
    global session_history
    session_history["is_active"] = True
    session_history["tension_count"] = 0
    session_history["total_frames"] = 0
    session_history["voice_scores"] = []
    session_history["start_blink"] = shared_data.get("blink_count", 0)
    return jsonify({"status": "Session Started", "duration": 60})


# ---------------- VOICE ANALYSIS ----------------
@app.route('/voice_analysis')
def voice_analysis():
    global voice_baseline, session_history
    try:
        if voice_baseline is None:
            return jsonify({'error': 'Calibrate first'}), 400

        audio, fs = record_audio(duration=3)
        if audio is None:
            return jsonify({'error': 'Mic error'}), 500

        current_stats = analyze_voice(audio, fs)
        voice_prob = calculate_lie_probability(current_stats, voice_baseline)
        
        if session_history["is_active"]:
            session_history["voice_scores"].append(voice_prob)
            session_history["total_frames"] += 1
            if shared_data.get("facial_prediction") == "Tense":
                session_history["tension_count"] += 1

        return jsonify({
            'voice_probability': voice_prob,
            'result': classify_voice_result(voice_prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- FINAL REPORT ----------------
@app.route('/get_session_report')
def get_session_report():
    global session_history
    session_history["is_active"] = False 
    
    total_blinks = shared_data.get("blink_count", 0) - session_history["start_blink"]

    tension_consistency = (
        session_history["tension_count"] / session_history["total_frames"] * 100
        if session_history["total_frames"] > 0 else 0
    )

    avg_voice_prob = (
        sum(session_history["voice_scores"]) / len(session_history["voice_scores"])
        if session_history["voice_scores"] else 0
    )

    final_prob, verdict = calculate_final_verdict(
        total_blinks, avg_voice_prob, tension_consistency
    )

    return jsonify({
        "verdict": verdict,
        "lie_probability": f"{final_prob}%",
        "blink_rate": f"{total_blinks} blinks/min",
        "voice_stress": f"{round(avg_voice_prob, 1)}%",
        "face_tension": f"{round(tension_consistency, 1)}%"
    })


# ---------------- BLINK COUNT ----------------
@app.route('/blink_count')
def blink_count():
    return jsonify(count=shared_data["blink_count"])


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)