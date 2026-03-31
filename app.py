import numpy as np
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
from face_test import generate_frames, shared_data
# Import the new functions from your updated voice.py
from voice import record_audio, analyze_voice, calculate_lie_probability, classify_voice_result

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Global storage for the "Normal" voice stats
voice_baseline = None 
users_db = {"admin": "admin"} # Simple in-memory storage for operators


# Login required decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# Home page is public
@app.route('/')
def home():
    return render_template('home.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple check, replace with real user validation
        if username in users_db and users_db[username] == password:
            session['logged_in'] = True
            return redirect(url_for('app_main'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users_db:
            error = 'Operator ID already exists.'
        else:
            users_db[username] = password
            session['logged_in'] = True
            return redirect(url_for('app_main'))
    return render_template('signup.html', error=error)

# Logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

# Main app page (protected)
@app.route('/app')
@login_required
def app_main():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NEW: CALIBRATION ROUTE ---
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

@app.route('/voice_analysis')
def voice_analysis():
    global voice_baseline
    try:
        if voice_baseline is None:
            return jsonify({'error': 'Please calibrate voice first!'}), 400

        # 1. Blink tracking
        start_blinks = shared_data.get("blink_count", 0)
        audio, fs = record_audio(duration=3)
        if audio is None:
            return jsonify({'error': 'Microphone error'}), 500

        end_blinks = shared_data.get("blink_count", 0)
        blinks_during_speech = end_blinks - start_blinks

        # 2. Voice Analysis (Relative to Baseline)
        current_stats = analyze_voice(audio, fs)
        # Using the new probability-based function
        voice_prob = calculate_lie_probability(current_stats, voice_baseline)
        
        # 3. Facial Result
        face_status = shared_data.get("facial_prediction", "Natural")

        # 4. Combined Logic (Final Verdict Score)
        # We start with the voice probability (0-100)
        total_lie_score = voice_prob
        
        # Add weight if face is tense
        if "Tense" in face_status:
            total_lie_score += 30 
        
        # Add weight if they blinked too much (Anxiety) or too little (Focusing on a lie)
        if blinks_during_speech > 6 or blinks_during_speech == 0:
            total_lie_score += 15

        # Final Classification
        final_result = "POSSIBLE LIE" if total_lie_score >= 60 else "LIKELY TRUTH"

        return jsonify({
            'pitch': float(round(current_stats[0], 2)),
            'energy': float(round(current_stats[1], 4)),
            'voice_probability': voice_prob,
            'face_prediction': face_status,
            'blinks_during_speech': blinks_during_speech,
            'total_score': total_lie_score,
            'result': final_result,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
    