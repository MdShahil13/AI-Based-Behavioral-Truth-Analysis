import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from blink import BlinkDetector
from facial_expression_lie_detector import predict_tension_from_facial_expression
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



shared_data = {"blink_count": 0, "facial_prediction": "Natural"}

def generate_frames():
    global shared_data
    shared_data["blink_count"] = 0
    blink_detector = BlinkDetector()
    cap = cv2.VideoCapture(0)
    # Move model loading here for faster camera startup
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    landmarker = FaceLandmarker.create_from_options(options)
    try:
        while cap.isOpened():
            timestamp_ms = int(time.time() * 1000)
            success, frame = cap.read()
            if not success:
                break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                
                blink_count = blink_detector.detect_blink(landmarks)
                shared_data["blink_count"] = blink_count

                # Convert MediaPipe landmarks to (x, y) tuples for lie detection
                landmark_points = [(lm.x, lm.y) for lm in landmarks]
                prediction = predict_tension_from_facial_expression(landmark_points)
                shared_data["facial_prediction"] = prediction
                
                
                h, w, _ = frame.shape
                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]

                # Draw the landmark dots
                for x, y in zip(xs, ys):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Define bounding box
                x1 = max(min(xs) - 20, 0)
                y1 = max(min(ys) - 20, 0)
                x2 = min(max(xs) + 20, w)
                y2 = min(max(ys) + 20, h)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display Lie/Truth prediction text only
                text = f'Status: {prediction}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if prediction == 'Tense' else (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()