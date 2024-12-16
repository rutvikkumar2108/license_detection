import cv2
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow
# import torch
# from matplotlib import pyplot as plt

# Load the model (change this line based on your model)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = YOLO('yolov8s.pt')
# Print the class names/labels
print(model.names)

# Path to the input video
input_video_path = "C:/Users/rutvi/Downloads/1721294-hd_1920_1080_25fps.mp4"
output_video_path = "C:/Users/rutvi/Desktop/annotated_video.mp4"

knife_id = 43
bicycle_id = 1
person_id = 0
car_id = 2
motorcycle_id = 3
Bus_id = 5
truck_id = 7

# Initialize video capture and writer
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class IDs and their corresponding labels
class_labels = {
    knife_id: "Knife",
    bicycle_id: "Bicycle",
    person_id: "Person",
    car_id: "Car",
    motorcycle_id: "Motorcycle",
    Bus_id: "Bus",
    truck_id: "Truck"
}

# Initialize counts for each class
class_counts = {label: 0 for label in class_labels.values()}

# Tracking detections to avoid multiple counts
tracked_objects = []  # List to hold unique detected objects


# Function to check if a detection is already tracked
def is_tracked(x1, y1, x2, y2, threshold=0.3):
    for (tx1, ty1, tx2, ty2) in tracked_objects:
        # Calculate IoU (Intersection over Union)
        xi1, yi1 = max(x1, tx1), max(y1, ty1)
        xi2, yi2 = min(x2, tx2), min(y2, ty2)
        inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (tx2 - tx1 + 1) * (ty2 - ty1 + 1)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area

        if iou > threshold:  # Considered as already tracked
            return True
    return False


# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Filter detections for specified classes
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)  # Convert class ID to integer
        if cls in class_labels:
            label = class_labels[cls]

            # Check if the object is already tracked
            if not is_tracked(x1, y1, x2, y2):
                # Add to tracked objects
                tracked_objects.append((x1, y1, x2, y2))

                # Increment the count for this class
                class_counts[label] += 1

                # Annotate the frame
                color = (0, 255, 0)  # Green for annotations
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2)

    # Write the annotated frame to the output video
    out.write(frame)

cap.release()
out.release()

# Print the counts for each class
print("Class Counts from Video:")
for label, count in class_counts.items():
    print(f"{label}: {count}")

# Display the output video
print(f"\nAnnotated video saved to: {output_video_path}")