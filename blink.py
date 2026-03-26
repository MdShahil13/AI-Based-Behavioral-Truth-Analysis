import math

class BlinkDetector:
    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.blink_count = 0
        self.blinking = False
    
    def eye_aspect_ratio(self, landmarks, eye_indices):
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]
        
        # Vertical distances
        v1 = math.dist((p2.x, p2.y), (p6.x, p6.y))
        v2 = math.dist((p3.x, p3.y), (p5.x, p5.y))
        # Horizontal distance
        h = math.dist((p1.x, p1.y), (p4.x, p4.y))
        
        if h == 0:
            return 1.0  # Avoid division by zero
        
        ear = (v1 + v2) / (2 * h)
        return ear
    
    def detect_blink(self, landmarks):
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.eye_aspect_ratio(landmarks, left_eye)
        right_ear = self.eye_aspect_ratio(landmarks, right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.threshold:
            if not self.blinking:
                self.blink_count += 1
                self.blinking = True
        else:
            self.blinking = False
        
        return self.blink_count