import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize the OCR engine
ocr = PaddleOCR(lang="en")

def extract_text_paddleocr(image):
    """
    Extract text from an image using PaddleOCR.

    Args:
        image: NumPy array representing the input image.

    Returns:
        List of extracted texts.
    """
    try:
        # Perform OCR on the input image (as a NumPy array)
        results = ocr.ocr(image, det=True, rec=True)  # Enable both detection and recognition
        texts = [result[1][0] for result in results[0]]  # Extract text from results
        return texts
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return []

# Load models
coco_model = YOLO('yolov8n.pt')  # COCO model for vehicle detection
license_plate_detector = YOLO('license_plate_detector.pt')  # Custom license plate detection model


def process_frame(frame, timestamp):
    """
    Process a video frame to detect license plates, draw bounding boxes, and return the results.

    Args:
        frame (numpy.ndarray): The video frame to process.
        timestamp (float): The timestamp of the frame in seconds.

    Returns:
        tuple: A list of detected plates with timestamps and the processed frame with bounding boxes.
    """
    # Initialize detected license plates
    detected_plates = []

    # Detect license plates in the frame
    license_plate_detections = license_plate_detector(frame)[0]

    for x1, y1, x2, y2, score, class_id in license_plate_detections.boxes.data.tolist():
        if score > 0.5:  # Confidence threshold
            # Crop the license plate from the frame
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            # Check if the cropped image is valid
            if license_plate_crop is not None and license_plate_crop.size > 0:
                # Extract text from the cropped license plate
                plate_text = extract_text_paddleocr(license_plate_crop)

                # Append detected plate information
                detected_plates.append({
                    'plate': plate_text[0] if plate_text else "None",
                    'timestamp': timestamp,
                    'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
                })

                # Draw a red bounding box for the license plate
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                if plate_text:
                    cv2.putText(frame, plate_text[0], (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return detected_plates, frame
