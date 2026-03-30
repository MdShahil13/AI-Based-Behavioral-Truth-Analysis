# facial_expression_lie_detector.py
"""
This module analyzes facial expressions to predict if a person is lying or not.
"""

def predict_tension_from_facial_expression(landmarks):
    """
    Predicts if a person is lying based on facial landmarks.
    Args:
        landmarks (list): List of facial landmark points.
    Returns:
        str: 'Lie' or 'Truth'
    """
    # Placeholder logic: Replace with actual ML model or rules
    # Example: If eyebrow raise and mouth corner pull detected, predict 'Lie'
    # This is a stub for demonstration purposes
    # Improved dummy logic: Use both high and low y values, and randomize a bit for demo
    if not landmarks or len(landmarks) < 400:
        return 'Truth'  # Not enough data, assume truth

    try:
        # MediaPipe indices: 159 (Left Eye Top), 52 (Left Eyebrow), 386 (Right Eye Top), 282 (Right Eyebrow)
        # Landmarks are (x, y), where y is normalized 0 to 1
        l_eye_y = landmarks[159][1]
        l_brow_y = landmarks[52][1]
        r_eye_y = landmarks[386][1]
        r_brow_y = landmarks[282][1]

        # Calculate relative distance between eye and eyebrow
        avg_dist = ((l_eye_y - l_brow_y) + (r_eye_y - r_brow_y)) / 2

        # Agar eyebrows 0.055 unit se zyada uthi hain, toh yeh tension (stress) dikhata hai
        if avg_dist > 0.055:
            return 'Tense'
    except Exception:
        return 'Natural'

    return 'Natural'

# Example usage (to be replaced with actual facial landmark extraction)
if __name__ == "__main__":
    # Dummy landmarks: list of (x, y) tuples
    sample_landmarks = [(0.1, 0.2), (0.3, 0.7), (0.4, 0.8), (0.5, 0.9), (0.2, 0.65), (0.6, 0.7), (0.7, 0.8)]
    result = predict_tension_from_facial_expression(sample_landmarks)
    print(f"Prediction: {result}")
