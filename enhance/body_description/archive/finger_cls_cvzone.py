from cvzone.HandTrackingModule import HandDetector
import cv2

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) 

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Determine which fingers are up
        raised_fingers = [name for i, name in enumerate(finger_names) if fingers[i] == 1]
        raised_fingers_str = ", ".join(raised_fingers) if raised_fingers else "None"

        # Display raised fingers
        cv2.rectangle(img, (10, 120), (600, 170), (0, 128, 0), -1)
        cv2.putText(img, f"Fingers Raised: {raised_fingers_str}", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
