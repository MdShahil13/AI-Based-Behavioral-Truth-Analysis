import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models.blink import BlinkDetector
from models.facial_expression import predict_tension_from_facial_expression

# ✅ Global Shared Data
shared_data = {"blink_count": 0, "facial_prediction": "Calibrating", "lie_probability": 0}

# ✅ Define 'detector' globally so app.py can find it
# Use IMAGE mode so it can process Base64 frames from the cloud
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# ✅ Persistent detector for blinks
blink_detector = BlinkDetector()

def analyze_frame(frame, session_history=None):
    """
    Processes a single frame sent from the browser.
    Replaces the old generate_frames() for cloud deployment.
    """
    global shared_data
    
    # 1. Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 2. Detect landmarks
    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks_raw = result.face_landmarks[0]
        landmark_points = [(lm.x, lm.y) for lm in landmarks_raw]
        
        # 3. Update Blinks
        shared_data["blink_count"] = blink_detector.detect_blink(landmarks_raw)

        # 4. Predict Tension (Using a standard baseline for now)
        prediction, _ = predict_tension_from_facial_expression(landmark_points, baseline_dist=0.05)
        shared_data["facial_prediction"] = prediction

        # 5. Update Session History if active
        if session_history and session_history.get("is_active"):
            session_history["total_frames"] += 1
            if "Tense" in prediction:
                session_history["tension_count"] += 1

    return shared_data
