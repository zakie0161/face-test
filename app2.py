import os
import json
import logging
import numpy as np
import cv2
import face_recognition

# Constants
IMAGE_FOLDER = "images"  # Folder containing images to process
KNOWN_FACES_FILE = "known_faces.json"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Global variable to store known faces
known_faces = []

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def save_known_faces():
    """Save known faces to a JSON file."""
    with open(KNOWN_FACES_FILE, "w") as file:
        json.dump(known_faces, file, default=lambda x: x.tolist())  # Convert NumPy arrays to lists
    logging.info("Saved known faces to file.")

def process_images_from_folder():
    """Process images from the IMAGE_FOLDER."""
    if not os.path.exists(IMAGE_FOLDER):
        logging.error(f"Image folder '{IMAGE_FOLDER}' does not exist.")
        return

    for filename in os.listdir(IMAGE_FOLDER):
        if allowed_file(filename):
            file_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                # Read and decode the image
                img = cv2.imread(file_path)
                if img is None:
                    logging.warning(f"Unable to read image: {file_path}")
                    continue

                # Convert to RGB and detect faces
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img_rgb)
                face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

                if not face_encodings:
                    logging.info(f"No faces detected in image: {file_path}")
                    continue

                # Compare detected faces with known faces
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        [np.array(f["encoding"]) for f in known_faces], face_encoding)
                    face_distances = face_recognition.face_distance(
                        [np.array(f["encoding"]) for f in known_faces], face_encoding)

                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        recognized_face = known_faces[best_match_index]
                        logging.info(f"Recognized {recognized_face['name']} in {filename} "
                                     f"with confidence {confidence:.2f}")
                    else:
                        logging.info(f"No match found for faces in {filename}")

            except Exception as e:
                logging.error(f"Error processing image '{file_path}': {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_known_faces()
    process_images_from_folder()
    save_known_faces()
