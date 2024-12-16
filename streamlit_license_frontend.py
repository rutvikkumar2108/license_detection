import streamlit as st
import cv2
import os
import numpy as np
from license_plate_detection import process_frame

def process_video(file_path):
    """
    Process the uploaded video and return frames with detected license plates.

    Args:
        file_path (str): Path to the uploaded video file.

    Returns:
        list: A list of dictionaries, each containing a frame, timestamp, and detected license plates.
    """
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // 5)  # Process approximately 5 FPS
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

        # Process the frame to detect license plates
        detected_plates, processed_frame = process_frame(frame, timestamp)

        # Append the frame, timestamp, and detected plates to results
        results.append({
            'frame': processed_frame,  # Processed frame with bounding boxes
            'timestamp': timestamp,
            'detected_plates': detected_plates
        })

    cap.release()
    return results

# Streamlit app layout
st.title("License Plate Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write("Processing the uploaded video...")
    with st.spinner("Analyzing..."):
        # Save the uploaded video temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the video and get the results
        results = process_video(file_path)
        st.success("Video processed successfully!")

        # Display the results
        for result in results:
            # Display the frame with timestamp
            st.image(result['frame'], caption=f"Frame at {result['timestamp']:.2f} seconds", channels="BGR")
            st.write(f"**Timestamp**: {result['timestamp']:.2f} seconds")

            # Display detected license plates
            if result['detected_plates']:
                st.write("**Detected License Plates**:")
                for plate in result['detected_plates']:
                    if isinstance(plate, dict) and 'plate' in plate:
                        st.write(f"License Plate: **{plate['plate']}**")
                    else:
                        st.write("Invalid plate data.")
            else:
                st.write("No license plates detected in this frame.")

            st.markdown("---")  # Divider between frames

        # Cleanup temporary video file
        os.remove(file_path)
