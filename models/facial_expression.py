"""
Analyzes facial landmarks to detect tension relative to a baseline.
"""

def predict_tension_from_facial_expression(landmarks, baseline_dist=None):
    if not landmarks or len(landmarks) < 400:
        return 'No Data', 0

    try:
        # MediaPipe Indices for Eyes and Brows
        l_eye_y = landmarks[159][1]
        l_brow_y = landmarks[52][1]
        r_eye_y = landmarks[386][1]
        r_brow_y = landmarks[282][1]

        # Calculate current vertical distance
        current_dist = ((l_eye_y - l_brow_y) + (r_eye_y - r_brow_y)) / 2

        # If we are calibrating, just return the distance
        if baseline_dist is None:
            return 'Calibrating', current_dist

        # TENSION LOGIC: 
        # Tension is detected if the distance shrinks by 20% or more from baseline
        # (e.g., if natural is 0.05, tension triggers at 0.04)
        threshold = baseline_dist * 0.80 

        if current_dist < threshold:
            return 'Tense', current_dist
        else:
            return 'Natural', current_dist

    except Exception as e:
        print("Error:", e)
        return 'Error', 0