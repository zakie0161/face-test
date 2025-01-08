import cv2
import numpy as np
import face_recognition
import threading
import logging
import os
import json
import queue
from concurrent.futures import ThreadPoolExecutor

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

class CameraStream:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url)
        self.frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._frame_grabber)
        self.thread.daemon = True
        self.thread.start()

    def _frame_grabber(self):
        """Grab frames from the camera and put them in the queue."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()  # Drop the oldest frame
                self.frame_queue.put(frame)

    def get_frame(self):
        """Retrieve a frame from the queue."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()

    def stop(self):
        """Stop the camera stream."""
        self.stop_event.set()
        self.thread.join()
        self.cap.release()

def process_ip_camera(camera_stream, window_name, frame_skip=5):
    frame_count = 0
    try:
        while True:
            frame = camera_stream.get_frame()
            if frame is None:
                continue

            # Skip frames for performance
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces and encode
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Use 'hog' for faster performance
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
        logging.error(f"Error in stream {window_name}: {e}")
    finally:
        cv2.destroyWindow(window_name)

def run_multi_ip_cameras(ip_camera_urls):
    camera_streams = []
    for url in ip_camera_urls:
        camera_streams.append(CameraStream(url))

    with ThreadPoolExecutor(max_workers=len(camera_streams)) as executor:
        for i, camera_stream in enumerate(camera_streams):
            window_name = f"Camera {i + 1}"
            executor.submit(process_ip_camera, camera_stream, window_name)

    # Wait for the threads to finish
    for camera_stream in camera_streams:
        camera_stream.stop()

if __name__ == "__main__":
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    load_known_faces()
    # List of IP camera stream URLs
    ip_camera_urls = [
        # "rtsp://admin:@WARMUP123@192.168.10.17:554/cam/realmonitor?channel=1&subtype=0",  # Replace with your IP camera URLs
        "rtsp://admin:@WARMUP123@192.168.10.17:554/cam/realmonitor?channel=8&subtype=0",  # Replace with your IP camera URLs
    ]
    run_multi_ip_cameras(ip_camera_urls)
