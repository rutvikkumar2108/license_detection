import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import base64
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# Initialize the OCR engine
ocr = PaddleOCR(lang="en")

VALID_STATE_CODES = [
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HR', 'HP', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN',
    'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS',
    'UK', 'UP', 'WB'
]

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
    Process a video frame to detect vehicles and license plates, draw bounding boxes, and return the results.

    Args:
        frame (numpy.ndarray): The video frame to process.
        timestamp (float): The timestamp of the frame in seconds.

    Returns:
        tuple: A dictionary of detected plates by vehicle type and the processed frame with bounding boxes.
    """
    # Convert numpy array (OpenCV) frame to PIL Image
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    height, width, _ = frame.shape

    # Detect vehicles using COCO model
    vehicle_classes = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }  # COCO class IDs for vehicles
    vehicle_detections = coco_model(frame)[0]

    # Group vehicle detections by class
    vehicle_boxes_by_type = {vehicle: [] for vehicle in vehicle_classes.values()}
    for x1, y1, x2, y2, score, class_id in vehicle_detections.boxes.data.tolist():
        if int(class_id) in vehicle_classes and score > 0.5:  # Confidence threshold
            vehicle_type = vehicle_classes[int(class_id)]
            vehicle_boxes_by_type[vehicle_type].append([int(x1), int(y1), int(x2), int(y2)])

            # Draw blue bounding box for vehicle
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline="red", width=2)

            # Load a smaller font for the vehicle type text
            font = ImageFont.load_default()  # You can use a custom font here if you prefer
            text = vehicle_type.capitalize()

            # Draw vehicle type text
            draw.text((int(x1), int(y1) - 10), text, fill="red", font=font)

    # Initialize detected license plates
    detected_plates_by_type = {vehicle: [] for vehicle in vehicle_classes.values()}

    # Detect license plates for each detected vehicle
    for vehicle_type, boxes in vehicle_boxes_by_type.items():
        for box in boxes:
            x1, y1, x2, y2 = box
            vehicle_crop = frame[y1:y2, x1:x2]  # Crop the frame to the vehicle region

            # Detect license plates within the cropped vehicle frame
            license_plate_detections = license_plate_detector(vehicle_crop)[0]
            for lx1, ly1, lx2, ly2, score, class_id in license_plate_detections.boxes.data.tolist():
                if score > 0.2:  # Confidence threshold
                    # Adjust license plate coordinates relative to the original frame
                    lx1 += x1
                    ly1 += y1
                    lx2 += x1
                    ly2 += y1

                    # Crop the license plate from the original frame
                    license_plate_crop = frame[int(ly1):int(ly2), int(lx1):int(lx2)]
                    if license_plate_crop is not None and license_plate_crop.size > 0:
                        # Extract text from the cropped license plate
                        plate_text = extract_text_paddleocr(license_plate_crop)
                        if plate_text:
                            detected_plates_by_type[vehicle_type].append({
                                'plate': plate_text[0],
                                'timestamp': timestamp
                            })

                            # Draw a red bounding box for the license plate
                            draw.rectangle([int(lx1), int(ly1), int(lx2), int(ly2)], outline="blue", width=2)

                            # Load a larger font for the license plate text
                            font_large = ImageFont.truetype("arial.ttf", size=20)  # Adjust the size as needed

                            # Draw the plate text with larger font
                            draw.text((int(lx1), int(ly2) + 20), plate_text[0], fill="blue", font=font_large)

            # If no license plate is detected, record the vehicle with an empty plate
            if not detected_plates_by_type[vehicle_type]:
                detected_plates_by_type[vehicle_type].append({
                    'plate': '',
                    'timestamp': timestamp
                })

    # Convert the PIL image back to a numpy array
    frame_with_boxes = np.array(pil_image)

    return detected_plates_by_type, frame_with_boxes


def validate_license_plates(plates, threshold=1):
    """
    Validate a list of license plates against the Indian license plate format.
    Add plates to the final list only if their occurrence exceeds the threshold.

    Args:
        plates (list): List of license plate dictionaries with `plate` and `timestamp`.
        threshold (int): Minimum occurrence for a plate to be considered valid.

    Returns:
        list: List of valid license plates with `plate` and `timestamp`.
    """
    # Define the regex pattern for Indian license plates
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"  # No spaces in the cleaned format

    # Clean and validate plates
    cleaned_plates = []
    for plate_info in plates:
        plate = plate_info['plate'].lower()
        # Remove spaces and validate format
        plate = plate.replace(" ", "")
        if re.match(pattern, plate.upper()):
            state_code = plate[:2].upper()
            if state_code in VALID_STATE_CODES:
                cleaned_plates.append(plate.upper())

    # Count occurrences of each plate
    plate_counts = Counter(cleaned_plates)

    # Filter plates based on threshold
    valid_plates = []
    for plate, count in plate_counts.items():
        if count > threshold:
            # Retain the first occurrence of the valid plate's timestamp
            for plate_info in plates:
                if plate_info['plate'].replace(" ", "").upper() == plate:
                    valid_plates.append({'plate': plate, 'timestamp': plate_info['timestamp']})
                    break

    return valid_plates


# def process_all_plates_by_type(all_plates_by_type, threshold=1):
#     """
#     Validate and clean license plates for each vehicle type in all_plates_by_type.
#
#     Args:
#         all_plates_by_type (dict): Dictionary containing lists of license plates for each vehicle type.
#         threshold (int): Minimum occurrence for a plate to be considered valid.
#
#     Returns:
#         dict: Dictionary with cleaned and validated license plates for each vehicle type.
#     """
#     validated_plates_by_type = {}
#     for vehicle_type, plates in all_plates_by_type.items():
#         validated_plates = validate_license_plates(plates, threshold)
#         validated_plates_by_type[vehicle_type] = validated_plates
#     return validated_plates_by_type

def process_all_plates_by_type(all_plates_by_type, threshold=1):
    """
    Validate and clean license plates for each vehicle type in all_plates_by_type.

    Args:
        all_plates_by_type (dict): Dictionary containing lists of license plates for each vehicle type.
                                   Each plate entry includes 'plate', 'timestamp', and 'frame'.
        threshold (int): Minimum occurrence for a plate to be considered valid.

    Returns:
        dict: Dictionary with cleaned and validated license plates for each vehicle type,
              including associated timestamps and frames.
    """
    validated_plates_by_type = {}

    for vehicle_type, plates in all_plates_by_type.items():
        # Validate plates and count occurrences
        validated_plates = validate_license_plates(plates, threshold)
        validated_plates_set = set(p['plate'] for p in validated_plates)

        # Retain validated plates with their associated frame and timestamp
        validated_plates_with_frames = []
        for plate_info in plates:
            if plate_info['plate'] in validated_plates_set:
                validated_plates_with_frames.append({
                    'plate': plate_info['plate'],
                    'timestamp': plate_info['timestamp'],
                    'frame': plate_info['frame']  # Include frame
                })
                validated_plates_set.remove(plate_info['plate'])  # Avoid duplicates

        validated_plates_by_type[vehicle_type] = validated_plates_with_frames

    return validated_plates_by_type

# # Read the video
# cap = cv2.VideoCapture("/C:/Users/rutvi/Pictures/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1) - Trim.mp4")
#
# # Get the video frame rate
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_skip = max(1, fps // 10)  # Skip frames to process approximately 5 FPS

# Process each frame and store detected number plates with timestamps
# all_plates_by_type = {'car': [], 'motorcycle': [], 'bus': [], 'truck': []}
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Skip frames
#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue
#
#     # Calculate timestamp for the current frame
#     timestamp = frame_count / fps
#
#     plates_by_type = process_frame(frame, timestamp)
#
#     # Append detected plates for each type
#     for vehicle_type, plates in plates_by_type.items():
#         all_plates_by_type[vehicle_type].extend(plates)
#
# # Remove duplicates based on plate number
# for vehicle_type in all_plates_by_type:
#     seen_plates = set()
#     unique_plates = []
#     for plate_info in all_plates_by_type[vehicle_type]:
#         if plate_info['plate'] not in seen_plates:
#             unique_plates.append(plate_info)
#             seen_plates.add(plate_info['plate'])
#     all_plates_by_type[vehicle_type] = unique_plates

# # Print results
# for vehicle_type, plates in all_plates_by_type.items():
#     print(f"{vehicle_type.capitalize()} Plates:")
#     for plate_info in plates:
#         print(f"  Plate: {plate_info['plate']}, Timestamp: {plate_info['timestamp']:.2f} seconds")



# Process and validate plates
# threshold = 2  # Example threshold
# validated_plates_by_type = process_all_plates_by_type(all_plates_by_type, threshold)
#
# # Print the results
# for vehicle_type, plates in validated_plates_by_type.items():
#     print(f"{vehicle_type.capitalize()} Valid Plates:")
#     for plate_info in plates:
#         print(f"  Plate: {plate_info['plate']}, Timestamp: {plate_info['timestamp']:.2f} seconds")
