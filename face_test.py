import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Crop (off-camera focus)")
    cv2.namedWindow("Full Frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)

            face_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw dots on full face
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            if face_crop.size > 0:
                cv2.imshow("Crop (off-camera focus)", face_crop)
        else:
            cv2.imshow("Crop (off-camera focus)", frame)

        cv2.imshow("Full Frame", frame)

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Crop (off-camera focus)", cv2.WND_PROP_VISIBLE) != 1:
            break

    cap.release()
    cv2.destroyAllWindows()