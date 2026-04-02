import numpy as np
import time
import base64
import cv2
import io
import librosa
import traceback
import os
import tempfile
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, session, g

# ✅ Models and Logic Imports
from models.face_test import analyze_frame, shared_data
from models.voice import analyze_voice, calculate_lie_probability
from models.result import calculate_final_verdict
from models.auth import auth
from models.db import users_collection

cv2.ocl.setUseOpenCL(False)  # Prevent OpenCL from trying to access GPU
cv2.setNumThreads(0)
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.register_blueprint(auth)

# --- GLOBAL STORAGE ---
voice_baseline = None 

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

@app.route('/app')
@login_required
def index():
    user_doc = users_collection.find_one({"email": session.get('user')})
    g.username = user_doc['username'] if user_doc else 'User'
    return render_template('index.html')

# ---------------- START SESSION ----------------
@app.route('/start_session')
def start_session():
    global session_history, shared_data
    
    shared_data["blink_count"] = 0 
    shared_data["facial_prediction"] = "Normal"

    session_history["is_active"] = True
    session_history["tension_count"] = 0
    session_history["total_frames"] = 0
    session_history["voice_scores"] = []
    session_history["start_blink"] = 0 
    
    return jsonify({"status": "Session Tracking Active and Reset"})

# ---------------- PROCESS FRAME ----------------
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global session_history, shared_data
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({"error": "Missing image key"}), 400

        img_str = data.get('image', '')

        if isinstance(img_str, str) and ',' in img_str:
            img_str = img_str.split(',')[1]

        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Decode failed"}), 400

        # ✅ Use centralized AI logic (MediaPipe handled inside)
        result = analyze_frame(frame, session_history)

        return jsonify({
            "status": "ok",
            "blinks": result["blink_count"],
            "expression": result["facial_prediction"]
        })

    except Exception as e:
        print(f"FRAME ERROR: {e}")
        return jsonify({"error": str(e)}), 400

# ---------------- CALIBRATE VOICE ----------------
@app.route('/calibrate_voice', methods=['POST'])
def calibrate_voice():
    global voice_baseline
    try:
        audio_file = request.files.get('audio_data')
        if not audio_file:
            return jsonify({'error': 'No data'}), 400

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")

        try:
            audio_file.save(temp_file.name)
            data, fs = librosa.load(temp_file.name, sr=None)

            voice_baseline = analyze_voice(data, fs)

            return jsonify({'status': 'Voice Calibrated'})
        finally:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- PROCESS FULL AUDIO ----------------
@app.route('/process_full_audio', methods=['POST'])
def process_full_audio():
    global voice_baseline, session_history
    try:
        audio_file = request.files.get('audio_data')
        if not audio_file or voice_baseline is None:
            return jsonify({'error': 'Missing calibration'}), 400

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")

        try:
            audio_file.save(temp_file.name)
            data, fs = librosa.load(temp_file.name, sr=None)

            current_stats = analyze_voice(data, fs)
            voice_prob = calculate_lie_probability(current_stats, voice_baseline)

            session_history["voice_scores"] = [voice_prob]

            return jsonify({'status': 'Success', 'prob': voice_prob})
        finally:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- FINAL REPORT ----------------
@app.route('/get_session_report')
def get_session_report():
    global session_history
    session_history["is_active"] = False 
    
    total_blinks = shared_data.get("blink_count", 0) - session_history["start_blink"]
    tension_percent = (
        (session_history["tension_count"] / session_history["total_frames"]) * 100
        if session_history["total_frames"] > 0 else 0
    )
    voice_prob = session_history["voice_scores"][0] if session_history["voice_scores"] else 0

    final_prob, verdict = calculate_final_verdict(
        total_blinks,
        voice_prob,
        tension_percent
    )

    return jsonify({
        "verdict": verdict,
        "lie_probability": f"{int(final_prob)}%",
        "blink_rate": f"{max(0, total_blinks)} blinks/min",
        "voice_stress": f"{round(voice_prob, 1)}%",
        "face_tension": f"{round(tension_percent, 1)}%"
    })

# ---------------- BLINK COUNT ----------------
@app.route('/blink_count')
def blink_count():
    return jsonify(count=shared_data["blink_count"])

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False, threaded=True)