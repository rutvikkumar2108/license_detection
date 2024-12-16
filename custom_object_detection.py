import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO COCO model
coco_model = YOLO('yolov8n.pt')  # COCO model for detection

def process_frame(frame, timestamp):
    """
    Process a video frame to detect persons, vehicles, and draw bounding boxes.

    Args:
        frame (numpy.ndarray): The video frame to process.
        timestamp (float): The timestamp of the frame in seconds.

    Returns:
        tuple: A dictionary of detected objects with timestamps and the processed frame with bounding boxes.
    """
    height, width, _ = frame.shape

    # COCO class IDs for relevant classes
    COCO_CLASSES = {
        0: 'Person',
        2: 'Car',
        3: 'Motorcycle',
        5: 'Bus',
        7: 'Truck'
    }

    # Colors for bounding boxes
    COLORS = {
        'Person': (255, 0, 0),      # Blue
        'Car': (0, 255, 0),         # Green
        'Motorcycle': (0, 255, 255),# Yellow
        'Bus': (255, 165, 0),       # Orange
        'Truck': (255, 0, 255)      # Magenta
    }

    # Detect objects using YOLO model
    detections = coco_model(frame)[0]

    # Initialize detected objects
    detected_objects = {class_name: [] for class_name in COCO_CLASSES.values()}

    # Iterate over detections and filter for relevant classes
    for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist():
        class_id = int(class_id)
        if class_id in COCO_CLASSES and score > 0.5:  # Confidence threshold
            class_name = COCO_CLASSES[class_id]

            # Append detection to the appropriate class list
            detected_objects[class_name].append({
                'timestamp': timestamp,
                'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
            })

            # Draw bounding box for the detected object
            color = COLORS[class_name]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, class_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detected_objects, frame
