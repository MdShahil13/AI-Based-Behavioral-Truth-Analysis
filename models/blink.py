import math

class BlinkDetector:
    def __init__(self, threshold=0.26):
        self.threshold = threshold
        self.blink_count = 0
        self.blinking = False
        self.frame_counter = 0;
    
    def eye_aspect_ratio(self, landmarks, eye_indices):
        try:
            # MediaPipe landmarks (p) have .x and .y attributes
            p = [landmarks[i] for i in eye_indices]
            v1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
            v2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
            h = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
            return (v1 + v2) / (2.0 * h) if h != 0 else 1.0
        except: 
            return 1.0
    
    def get_current_ear(self, landmarks):
        # MediaPipe Face Mesh standard indices
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        l_ear = self.eye_aspect_ratio(landmarks, left_eye)
        r_ear = self.eye_aspect_ratio(landmarks, right_eye)
        return (l_ear + r_ear) / 2.0

    # Renamed this to match what your app.py is calling
    def detect_blink(self, landmarks):
        if not landmarks: 
            return self.blink_count
            
        ear = self.get_current_ear(landmarks)
        
        if ear < self.threshold:
            if not self.blinking:
                self.blink_count += 1
                self.blinking = True
                print(f"Blink Detected! Count: {self.blink_count}")
        else: 
            self.blinking = False
            
        return self.blink_count

    def set_threshold(self, natural_ear):
        self.threshold = natural_ear * 0.70
