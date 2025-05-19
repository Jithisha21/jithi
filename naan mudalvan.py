import cv2
import numpy as np

 # Load video (or use 0 for webcam)
cap = cv2.VideoCapture('bheem.mp4')  # or use 0

 # Define color range for object (example: yellow fish or coral)
lower_color = np.array([20, 100, 100])   # HSV lower bound
upper_color = np.array([40, 255, 255])   # HSV upper bound

while True:
     ret, frame = cap.read()
     if not ret:
         break

     # Resize for speed
     frame = cv2.resize(frame, (640, 480))

     # Convert to HSV
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # Create mask
     mask = cv2.inRange(hsv, lower_color, upper_color)

     # Morphological filtering to reduce noise
     kernel = np.ones((5, 5), np.uint8)
     mask = cv2.erode(mask, kernel, iterations=1)
     mask = cv2.dilate(mask, kernel, iterations=2)

     # Find contours
     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     # Draw bounding boxes
     for cnt in contours:
         area = cv2.contourArea(cnt)
         if area > 500:
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
             cv2.putText(frame, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

     # Show results
     cv2.imshow("Underwater View", frame)
     cv2.imshow("Mask", mask)

     if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
         break

cap.release()
cv2.destroyAllWindows()