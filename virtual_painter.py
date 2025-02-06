import cv2
import numpy as np
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height

    # Initialize hand detector
    detector = HandDetector()

    # Drawing variables
    draw_color = (255, 0, 255)  # Pink color
    brush_thickness = 15
    eraser_thickness = 100
    
    # Create canvas
    canvas = np.zeros((720, 1280, 3), np.uint8)
    
    # Previous coordinates for drawing
    px, py = 0, 0
    
    while True:
        # Get image from webcam
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Mirror image

        # Find hand landmarks
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            # Index finger tip coordinates
            x1, y1 = lm_list[8][1:]
            # Middle finger tip coordinates
            x2, y2 = lm_list[12][1:]

            # Check which fingers are up
            fingers = []
            if lm_list[8][2] < lm_list[6][2]:  # Index finger
                fingers.append(1)
            else:
                fingers.append(0)
            if lm_list[12][2] < lm_list[10][2]:  # Middle finger
                fingers.append(1)
            else:
                fingers.append(0)

            # Drawing Mode - Index finger up, middle finger down
            if fingers[0] == 1 and fingers[1] == 0:
                if px == 0 and py == 0:
                    px, py = x1, y1
                cv2.line(canvas, (px, py), (x1, y1), draw_color, brush_thickness)
                cv2.circle(img, (x1, y1), brush_thickness//2, draw_color, cv2.FILLED)
                px, py = x1, y1

            # Eraser Mode - Both index and middle fingers up
            elif fingers[0] == 1 and fingers[1] == 1:
                if px == 0 and py == 0:
                    px, py = x1, y1
                cv2.line(canvas, (px, py), (x1, y1), (0, 0, 0), eraser_thickness)
                px, py = x1, y1
            else:
                px, py = 0, 0

        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        cv2.imshow("Image", img)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
