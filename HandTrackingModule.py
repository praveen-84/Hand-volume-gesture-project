
import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Tip IDs for fingers
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """Detects hands in the image and optionally draws landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Finds landmark positions and returns a list of landmarks and bounding box."""
        x_list = []
        y_list = []
        bbox = []
        lm_list = []

        # Error handling in case no hands are detected
        if self.results.multi_hand_landmarks:
            try:
                my_hand = self.results.multi_hand_landmarks[hand_no]
            except IndexError:
                print("Hand number out of range.")
                return lm_list, bbox
            
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                lm_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            # Calculate the bounding box around the hand
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax
            
            # Draw bounding box
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return lm_list, bbox

    def fingers_up(self):
        """Determines which fingers are up."""
        fingers = []
        
        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        """Finds the distance between two landmarks and optionally draws connecting lines."""
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Optional drawing
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        # Calculate distance
        length = math.hypot(x2 - x1, y2 - y1)
        
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    p_time = 0
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use index 0 for the default camera
    
    # Initialize the hand detector
    detector = HandDetector()
    
    # Main loop
    while True:
        success, img = cap.read()
        
        # Check if the frame capture was successful
        if not success:
            print("Failed to capture frame.")
            break
        
        # Find hands in the image
        img = detector.find_hands(img)
        
        # Find positions and bounding box of landmarks
        lm_list, bbox = detector.find_position(img)
        
        # Output landmark list and bounding box
        if lm_list:
            print("Landmark list:", lm_list)
            print("Bounding box:", bbox)
        
        # Calculate the current time and frames per second (FPS)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Display FPS on the image
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        
        # Show the image with drawn landmarks and FPS
        cv2.imshow("Image", img)
        
        # Wait for a key press to end the loop, or press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
