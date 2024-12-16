import cv2
import numpy as np
from ultralytics import YOLO
coco_model = YOLO('yolov8n.pt')  # COCO model for vehicle detection
def process_frame(frame, timestamp):
    """
    Process a video frame to detect persons, draw bounding boxes, and return the results.

    Args:
        frame (numpy.ndarray): The video frame to process.
        timestamp (float): The timestamp of the frame in seconds.

    Returns:
        tuple: A list of detected persons with timestamps and the processed frame with bounding boxes.
    """
    height, width, _ = frame.shape

    # COCO class ID for persons
    PERSON_CLASS_ID = 0  # COCO class ID for "person"

    # Detect objects using COCO model
    detections = coco_model(frame)[0]

    # Initialize detected persons
    detected_persons = []

    # Iterate over detections and filter for persons
    for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist():
        if int(class_id) == PERSON_CLASS_ID and score > 0.5:  # Confidence threshold
            # Append person detection to the list
            detected_persons.append({
                'timestamp': timestamp,
                'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
            })

            # Draw blue bounding box for person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return detected_persons, frame
