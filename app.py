import numpy as np
import time
import base64
import cv2
from functools import wraps
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session, g
from models.face_test import shared_data
from models.voice import record_audio, analyze_voice, calculate_lie_probability, classify_voice_result
from models.result import calculate_final_verdict
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import mediapipe as mp

# ✅ Models and Logic Imports
from models.face_test import shared_data, detector, blink_detector # Using your detector instance
from models.voice import record_audio, analyze_voice, calculate_lie_probability, classify_voice_result
from models.result import calculate_final_verdict
from models.blink import BlinkDetector # Ensure this is imported
from models.facial_expression import predict_tension_from_facial_expression


# ✅ Import Blueprint
from models.auth import auth

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.register_blueprint(auth)

# --- GLOBAL STORAGE ---
voice_baseline = None 
blink_detector = BlinkDetector() # Initialize detector

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


from models.db import users_collection

@app.route('/app')
@login_required
def index():
    # Find username from session['user'] (which is email)
    user_doc = users_collection.find_one({"email": session.get('user')})
    g.username = user_doc['username'] if user_doc else 'User'
    return render_template('index.html')

# ---------------- NEW: PROCESS FRAME (FOR DEPLOYMENT) ----------------

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global session_history, shared_data
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400
    
    try:
        # Correctly reference the string variable
        base64_str = data['image'] 
        img_data = base64.b64decode(base64_str.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None: return jsonify({"error": "Invalid frame"}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = detector.detect(mp_image)

        # Check if landmarks exist in the result object
        if result and result.face_landmarks:
            # result.face_landmarks is a list of lists of landmark objects
            landmarks_raw = result.face_landmarks[0]
            
            # Update Blink Count
            shared_data["blink_count"] = blink_detector.detect_blink(landmarks_raw)

            # Update Facial Tension
            landmark_points = [(lm.x, lm.y) for lm in landmarks_raw]
            prediction, _ = predict_tension_from_facial_expression(landmark_points, baseline_dist=0.05)
            shared_data["facial_prediction"] = prediction

            if session_history.get("is_active"):
                session_history["total_frames"] += 1
                if "Tense" in prediction:
                    session_history["tension_count"] += 1

        return jsonify({
            "blinks": shared_data["blink_count"],
            "prediction": shared_data.get("facial_prediction", "Normal")
        })

    except Exception as e:
        print(f"Process Error: {e}")
        return jsonify({"error": str(e)}), 400


# ---------------- CALIBRATE VOICE ----------------
@app.route('/calibrate_voice')
def calibrate_voice():
    global voice_baseline
    try:
        audio, fs = record_audio(duration=3)
        if audio is not None:
            voice_baseline = analyze_voice(audio, fs)
            # Ensure index 0 for pitch
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
        if audio is None: return jsonify({'error': 'Mic error'}), 500

        current_stats = analyze_voice(audio, fs)
        voice_prob = calculate_lie_probability(current_stats, voice_baseline)
        
        if session_history["is_active"]:
            session_history["voice_scores"].append(voice_prob)

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
    tension_consistency = (session_history["tension_count"] / session_history["total_frames"] * 100) if session_history["total_frames"] > 0 else 0
    avg_voice_prob = sum(session_history["voice_scores"]) / len(session_history["voice_scores"]) if session_history["voice_scores"] else 0

    final_prob, verdict = calculate_final_verdict(total_blinks, avg_voice_prob, tension_consistency)

    return jsonify({
        "verdict": verdict,
        "lie_probability": f"{final_prob}%",
        "blink_rate": f"{total_blinks} blinks/min",
        "voice_stress": f"{round(avg_voice_prob, 1)}%",
        "face_tension": f"{round(tension_consistency, 1)}%"
    })

@app.route('/blink_count')
def blink_count():
    return jsonify(count=shared_data["blink_count"])

if __name__ == "__main__":
    # Note: Deployed apps often need 0.0.0.0 to be accessible
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
