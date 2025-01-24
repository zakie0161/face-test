import cv2 # type: ignore
import numpy as np
import threading
import os
import face_recognition

# Function for face detection on a single video stream
def process_video(video_path, window_name):
    # Load the DNN face detection model with CUDA support
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA backend
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # Use CUDA target
    
    output_dir = "detected_faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame using CUDA for better performance
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)  # Upload frame to GPU
        gpu_frame_resized = cv2.cuda.resize(gpu_frame, (640, 480))  # Resize in GPU (higher resolution)
        frame_resized = gpu_frame_resized.download()  # Download the resized frame back to CPU

        # Convert resized frame to blob suitable for DNN processing
        blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 177, 123), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Get the frame dimensions
        (h, w) = frame_resized.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Increased confidence threshold for better accuracy
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face = frame_resized[startY:endY, startX:endX]
                img_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img_rgb)
                face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
                if(face_encodings):
                    print("found")
                face_filename = os.path.join(output_dir, f"face_{saved_count:04d}.jpg")
                # cv2.imwrite(face_filename, face)
                # saved_count += 1


        # Display the frame with detected faces
        cv2.imshow(window_name, frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Function to start multiple video processing threads
def start_video_processing(video_paths):
    threads = []
    
    # Create and start a thread for each video file
    for i, video_path in enumerate(video_paths):
        window_name = f"Video {i+1}"
        thread = threading.Thread(target=process_video, args=(video_path, window_name))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

# List of video files you want to process
video_paths = [
    "video.mp4", 
    # "rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=7&subtype=0",
    # "rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=8&subtype=0",
    # "rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=2&subtype=0"
    # "rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=0"
    ]

# Start processing the videos concurrently
start_video_processing(video_paths)
