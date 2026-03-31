import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from blink import BlinkDetector
from facial_expression_lie_detector import predict_tension_from_facial_expression

# Import the session tracker from your app file
# Note: This requires 'session_history' to be defined in your app.py
try:
    from app import session_history
except ImportError:
    # Fallback if importing from app fails during standalone testing
    session_history = {"is_active": False, "tension_count": 0, "total_frames": 0}

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

shared_data = {"blink_count": 0, "facial_prediction": "Calibrating", "lie_probability": 0}

def generate_frames():
    global shared_data, session_history
    blink_detector = BlinkDetector()
    cap = cv2.VideoCapture(0)
    
    # --- CALIBRATION CONFIG ---
    calibration_frames = 100 
    frame_counter = 0
    baseline_distances = []
    baseline_ears = []  
    final_baseline_tension = None

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    
    landmarker = FaceLandmarker.create_from_options(options)

    try:
        while cap.isOpened():
            timestamp_ms = int(time.time() * 1000)
            success, frame = cap.read()
            if not success: break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks_raw = result.face_landmarks[0]
                landmark_points = [(lm.x, lm.y) for lm in landmarks_raw]
                
                # 1. BLINK DETECTION
                shared_data["blink_count"] = blink_detector.detect_blink(landmarks_raw)

                # 2. CALIBRATION VS PREDICTION
                if frame_counter < calibration_frames:
                    _, current_dist = predict_tension_from_facial_expression(landmark_points)
                    baseline_distances.append(current_dist)
                    
                    current_ear = blink_detector.get_current_ear(landmarks_raw) if hasattr(blink_detector, 'get_current_ear') else 0.3
                    baseline_ears.append(current_ear)
                    
                    prediction = f"Learning Face: {int((frame_counter/calibration_frames)*100)}%"
                    frame_counter += 1
                else:
                    if final_baseline_tension is None:
                        final_baseline_tension = np.mean(baseline_distances)
                        if hasattr(blink_detector, 'set_threshold'):
                            blink_detector.set_threshold(np.mean(baseline_ears))
                    
                    prediction, _ = predict_tension_from_facial_expression(landmark_points, final_baseline_tension)

                # --- 3. SESSION TRACKING LOGIC (ADDED HERE) ---
                if session_history.get("is_active"):
                    session_history["total_frames"] += 1
                    if "Tense" in prediction:
                        session_history["tension_count"] += 1
                # ----------------------------------------------

                shared_data["facial_prediction"] = prediction

                # --- VISUALS ---
                h, w, _ = frame.shape
                for i, lm in enumerate(landmarks_raw):
                    if i % 15 == 0: 
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                
                color = (255, 255, 255)
                if "Natural" in prediction: color = (0, 255, 0)
                if "Tense" in prediction: color = (0, 0, 255)

                # Display "SESSION RECORDING" if active
                if session_history.get("is_active"):
                    cv2.putText(frame, "● RECORDING SESSION", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, f"STATUS: {prediction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"BLINKS: {shared_data['blink_count']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    finally:
        cap.release()