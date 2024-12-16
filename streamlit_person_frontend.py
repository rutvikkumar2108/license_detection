import streamlit as st
import cv2
import os
from person_detection import process_frame

def process_video(file_path):
    """
    Process the uploaded video to detect persons and return frames with bounding boxes and timestamps.

    Args:
        file_path (str): Path to the uploaded video file.

    Returns:
        list: A list of dictionaries, each containing a frame, timestamp, and detected persons.
    """
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // 2)  # Process approximately 5 FPS
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

        # Process the frame and get detected persons and the processed frame
        detected_persons, processed_frame = process_frame(frame, timestamp)

        # Append the frame, timestamp, and detected persons to results
        results.append({
            'frame': processed_frame,
            'timestamp': timestamp,
            'detected_persons': detected_persons
        })

    cap.release()
    return results

# Streamlit app layout
st.title("Person Detection in Video")

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
            # Display the processed frame
            st.image(result['frame'], caption=f"Frame at {result['timestamp']:.2f} seconds", channels="BGR")
            st.write(f"**Timestamp**: {result['timestamp']:.2f} seconds")

            # List detected persons in the frame
            if result['detected_persons']:
                st.write("Detected Persons:")
                for person in result['detected_persons']:
                    st.write(f"Bounding Box: {person['bounding_box']}")
            else:
                st.write("No persons detected in this frame.")

            st.markdown("---")  # Divider between frames

        # Cleanup temporary video file
        os.remove(file_path)
