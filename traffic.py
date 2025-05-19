import cv2
import numpy as np

def detect_traffic_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red mask
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # Yellow mask
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Green mask
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # List to store detected regions
    detected_lights = []

    # Function to find and annotate traffic lights
    def find_and_annotate(mask, color, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Threshold to filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                detected_lights.append((x, y, w, h, label))  # Store the light region

    # Detect and annotate red lights
    find_and_annotate(red_mask, (0, 0, 255), 'Red Light')
    # Detect and annotate yellow lights
    find_and_annotate(yellow_mask, (0, 255, 255), 'Yellow Light')
    # Detect and annotate green lights
    find_and_annotate(green_mask, (0, 255, 0), 'Green Light')

    # Return the image with annotated traffic lights
    return frame, detected_lights

if _name_ == "_main_":
    image_path = ".jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image")
    else:
        output, detected_lights = detect_traffic_lights(frame)
        cv2.imshow("Traffic Light Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the output image with annotations
        output_image_path = "output_traffic_lights.jpg"
        cv2.imwrite(output_image_path, output)
        print(f"Annotated image saved to {output_image_path}")