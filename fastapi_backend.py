from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from car_license_plate_detection import process_all_plates_by_type, process_frame
import cv2
import logging

app = FastAPI()
# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upload_video")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded file
    logger.info(f"Received file: {file.filename}")
    file_path = f"temp_{file.filename}"
    print('1')
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video (Assuming process_all_plates_by_type uses a video file)
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // 5)  # Skip frames to process approximately 5 FPS
    frame_count = 0
    all_plates_by_type = {'car': [], 'motorcycle': [], 'bus': [], 'truck': []}
    print("2")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        timestamp = frame_count / fps
        print('3')
        # Assuming process_frame is imported from the same script
        plates_by_type, processed_frame = process_frame(frame, timestamp)
        for vehicle_type, plates in plates_by_type.items():
            # all_plates_by_type[vehicle_type].extend(plates)
            for plate_info in plates:
                all_plates_by_type[vehicle_type].append({
                    'plate': plate_info['plate'],
                    'timestamp': plate_info['timestamp'],
                    'frame': processed_frame  # Store the frame for this plate
                })
                print('4')

    cap.release()
    # os.remove(file_path)

    # # Remove duplicates
    # for vehicle_type in all_plates_by_type:
    #     seen_plates = set()
    #     unique_plates = []
    #     for plate_info in all_plates_by_type[vehicle_type]:
    #         if plate_info['plate'] not in seen_plates:
    #             unique_plates.append(plate_info)
    #             seen_plates.add(plate_info['plate'])
    #     all_plates_by_type[vehicle_type] = unique_plates

    # Remove duplicates and keep associated frames
    print('5')
    for vehicle_type in all_plates_by_type:
        seen_plates = set()
        unique_plates = []
        for plate_info in all_plates_by_type[vehicle_type]:
            plate_key = plate_info['plate']
            if plate_key not in seen_plates:
                unique_plates.append(plate_info)
                seen_plates.add(plate_key)
        all_plates_by_type[vehicle_type] = unique_plates
    print('6')
    validated_plates_by_type = process_all_plates_by_type(all_plates_by_type, threshold=2)

    return JSONResponse(validated_plates_by_type)

#
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import shutil
# import os
# import cv2
# from car_license_plate_detection import process_frame, process_all_plates_by_type
#
# app = FastAPI()
#
# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)):
#     # Save the uploaded file
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#
#     # Open the video file
#     cap = cv2.VideoCapture(file_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_skip = max(1, fps // 10)  # Process approximately 5 FPS
#     frame_count = 0
#     all_plates_by_type = {'car': [], 'motorcycle': [], 'bus': [], 'truck': []}
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Skip frames
#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue
#
#         timestamp = frame_count / fps
#
#         # Process the frame and get plates_by_type and the processed frame
#         plates_by_type, processed_frame = process_frame(frame, timestamp)
#
#         # Append detected plates along with the processed frame
#         for vehicle_type, plates in plates_by_type.items():
#             for plate_info in plates:
#                 all_plates_by_type[vehicle_type].append({
#                     'plate': plate_info['plate'],
#                     'timestamp': plate_info['timestamp'],
#                     'frame': processed_frame  # Store the frame for this plate
#                 })
#
#     cap.release()
#     os.remove(file_path)
#
#     # Remove duplicates and keep associated frames
#     for vehicle_type in all_plates_by_type:
#         seen_plates = set()
#         unique_plates = []
#         for plate_info in all_plates_by_type[vehicle_type]:
#             plate_key = plate_info['plate']
#             if plate_key not in seen_plates:
#                 unique_plates.append(plate_info)
#                 seen_plates.add(plate_key)
#         all_plates_by_type[vehicle_type] = unique_plates
#
#     # Validate plates and retain associated frames
#     validated_plates_by_type = {}
#     for vehicle_type, plates in all_plates_by_type.items():
#         validated_plates = process_all_plates_by_type({vehicle_type: plates}, threshold=2)
#         if vehicle_type in validated_plates:
#             validated_plates_by_type[vehicle_type] = [
#                 {
#                     'plate': plate_info['plate'],
#                     'timestamp': plate_info['timestamp'],
#                     'frame': plate_info['frame']  # Include frame in the response
#                 }
#                 for plate_info in plates
#                 if plate_info['plate'] in [p['plate'] for p in validated_plates[vehicle_type]]
#             ]
#
#     # Return the validated plates with associated frames
#     return JSONResponse(validated_plates_by_type)
