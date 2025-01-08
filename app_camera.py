import cv2
import numpy as np
import face_recognition
import threading
import logging
import os
import json

IMAGE_FOLDER = "images"
KNOWN_FACES_FILE = "known_faces.json"

# Preload known faces
known_faces = []

def load_known_faces():
    """Load known faces from a JSON file."""
    global known_faces
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "r") as file:
            known_faces = json.load(file)
            for face in known_faces:
                face["encoding"] = np.array(face["encoding"])  # Convert encoding back to NumPy array
        logging.info("Loaded known faces from file.")
    else:
        logging.info("No known faces file found. Starting fresh.")

# Process frames from an IP camera
def process_ip_camera(stream_url, window_name, frame_skip=5):
    video_capture = cv2.VideoCapture(0)
    frame_count = 0

    if not video_capture.isOpened():
        print(f"Error: Unable to access stream {stream_url}")
        return

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Failed to grab frame from {stream_url}")
                break

            # Skip frames for performance
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces and encode
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    [np.array(f["encoding"]) for f in known_faces], face_encoding)
                face_distances = face_recognition.face_distance(
                    [np.array(f["encoding"]) for f in known_faces], face_encoding)

                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]

                    # Draw rectangle and label
                    top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    name = known_faces[best_match_index]["name"]
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Unknown face
                    top, right, bottom, left = [v * 4 for v in face_location]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show the video feed
            cv2.imshow(window_name, frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Error in stream {stream_url}: {e}")
    finally:
        video_capture.release()
        cv2.destroyWindow(window_name)

# Run multiple IP cameras
def run_multi_ip_cameras(ip_camera_urls):
    threads = []

    for i, stream_url in enumerate(ip_camera_urls):
        window_name = f"Camera {i+1}"
        thread = threading.Thread(target=process_ip_camera, args=(stream_url, window_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    load_known_faces()
    # List of IP camera stream URLs
    ip_camera_urls = [
        # "rtsp://admin:@WARMUP123@192.168.10.17:554/cam/realmonitor?channel=1&subtype=0",  # Replace with your IP camera URLs
        "rtsp://admin:@WARMUP123@192.168.10.17:554/cam/realmonitor?channel=8&subtype=0",  # Replace with your IP camera URLs
    ]
    run_multi_ip_cameras(ip_camera_urls)
