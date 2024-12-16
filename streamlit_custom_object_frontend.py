import streamlit as st
import cv2
import os
from custom_object_detection import process_frame

def process_video(file_path):
    """
    Process the uploaded video to detect objects (persons, vehicles) and return frames with timestamps.

    Args:
        file_path (str): Path to the uploaded video file.

    Returns:
        list: A list of dictionaries, each containing a frame, timestamp, and detected objects.
    """
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // 2)  # Process approximately every second frame
    frame_count = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        timestamp = frame_count / fps

        # Process the frame and get detected objects and the processed frame
        detected_objects, processed_frame = process_frame(frame, timestamp)

        # Append the frame and timestamp to results if any relevant objects are detected
        if any(detected_objects.values()):  # Check if there are any detections
            results.append({
                'frame': processed_frame,
                'timestamp': timestamp
            })

    cap.release()
    return results

# Streamlit app layout
st.title("Object Detection in Video")

# File uploader for video files
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write("Processing the uploaded video...")
    with st.spinner("Analyzing..."):
        # Save the uploaded video temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the video
        results = process_video(file_path)
        st.success("Video processed successfully!")

        # Display results
        for result in results:
            # Display the processed frame and timestamp
            st.image(result['frame'], caption=f"Frame at {result['timestamp']:.2f} seconds", channels="BGR")
            st.write(f"**Timestamp**: {result['timestamp']:.2f} seconds")

            st.markdown("---")  # Divider between frames

        # Cleanup temporary video file
        os.remove(file_path)
