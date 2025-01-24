from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
import uuid
import json
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
IMAGE_FOLDER = "images"
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


@app.route("/recognize", methods=["POST"])
def recognize_faces():
    """Recognize faces in an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"data": None, "code": 400, "message": "No image file provided"}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({"data": None, "code": 400, "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed."}), 400

    # Optional: Get the name filter from form data
    name_filter = request.form.get('name', None)

    try:
        # Read and decode the image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"data": None, "code": 400, "message": "Unable to decode image file"}), 400

        # Convert to RGB and detect faces
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if not face_encodings:
            return jsonify({"data": None, "code": 404, "message": "No face detected"}), 404

        # If a name filter is provided, filter known faces by the name
        filtered_known_faces = (
            [f for f in known_faces if f["name"].lower() == name_filter.lower()]
            if name_filter else known_faces
        )

        if not filtered_known_faces:
            return jsonify({"data": None, "code": 404, "message": f"No known faces match"}), 404

        # Compare detected faces with known faces
        best_match = None
        highest_confidence = 0

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                [np.array(f["encoding"]) for f in filtered_known_faces], face_encoding)
            face_distances = face_recognition.face_distance(
                [np.array(f["encoding"]) for f in filtered_known_faces], face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = {
                        "name": filtered_known_faces[best_match_index]["name"],
                        "path": filtered_known_faces[best_match_index]["path"],
                        "confidence": round(confidence, 2),
                    }

        if best_match:
            return jsonify({"data": best_match, "code": 200, "message": "Success"})
        else:
            return jsonify({"data": None, "code": 404, "message": "No known faces recognized in the image"}), 404

    except Exception as e:
        logging.error(f"Error in recognize_faces: {e}")
        return jsonify({"data": None, "code": 500, "error": "Failed to process image", "details": str(e)}), 500


@app.route("/faces", methods=["GET", "POST", "PUT", "DELETE"])
def manage_faces():
    """CRUD operations for known faces."""
    global known_faces

    if request.method == "GET":
        # Return the list of known faces
        return jsonify({"data": [{"name": f["name"], "path": f["path"]} for f in known_faces], "code": 200, "message": "Success"})

    elif request.method == "POST":
        # Add a new known face
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({"data": None, "code": 400, "message": "Image file and name are required"}), 400

        file = request.files['image']
        name = request.form['name']

        if not allowed_file(file.filename):
            return jsonify({"data": None, "code": 400, "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed."}), 400

        try:
            # Save the uploaded image
            file_extension = file.filename.rsplit('.', 1)[-1].lower()
            random_filename = f"{uuid.uuid4()}.{file_extension}"
            save_path = os.path.join(IMAGE_FOLDER, random_filename)
            file.save(save_path)

            # Process the saved image for face recognition
            img = cv2.imread(save_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)

            if not encodings:
                os.remove(save_path)
                return jsonify({"data": None, "code": 400, "message": "No face detected in the image"}), 404

            known_faces.append({
                "name": name,
                "encoding": encodings[0],
                "path": save_path
            })
            save_known_faces()

            return jsonify({"data": None, "code": 200, "message": f"Added {name} to known faces", "path": save_path}), 201

        except Exception as e:
            logging.error(f"Error in POST /faces: {e}")
            return jsonify({"data": None, "code": 400, "message": "Failed to add face", "details": str(e)}), 500

    elif request.method == "PUT":
        # Update an existing face
        if 'name' not in request.form:
            return jsonify({"data": None, "code": 400, "message": "Name is required to update a face"}), 400

        name = request.form['name']
        new_name = request.form.get('new_name')  # Optional
        new_image = request.files.get('image')  # Optional

        face_record = next((f for f in known_faces if f["name"] == name), None)
        if not face_record:
            return jsonify({"data": None, "code": 400, "message": f"No face found with name '{name}'"}), 404

        try:
            if new_name:
                face_record["name"] = new_name

            if new_image:
                file_extension = new_image.filename.rsplit('.', 1)[-1].lower()
                random_filename = f"{uuid.uuid4()}.{file_extension}"
                new_save_path = os.path.join(IMAGE_FOLDER, random_filename)
                new_image.save(new_save_path)

                img = cv2.imread(new_save_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img_rgb)

                if not encodings:
                    os.remove(new_save_path)
                    return jsonify({"data": None, "code": 404, "message": "No face detected in the new image"}), 404

                os.remove(face_record["path"])  # Remove the old image
                face_record["encoding"] = encodings[0]
                face_record["path"] = new_save_path

            save_known_faces()
            return jsonify({"data": name, "code": 200, "message": f"Updated face for '{name}'"}), 200

        except Exception as e:
            logging.error(f"Error in PUT /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to update face", "details": str(e)}), 500

    elif request.method == "DELETE":
        if 'name' not in request.args:
            return jsonify({"data": None, "code": 400, "message": "Name is required to delete a face"}), 400

        name = request.args['name']
        face_record = next((f for f in known_faces if f["name"] == name), None)

        if not face_record:
            return jsonify({"data": None, "code": 404, "message": f"No face found with name '{name}'"}), 404

        try:
            os.remove(face_record["path"])
            known_faces.remove(face_record)
            save_known_faces()
            return jsonify({"data": name, "code": 200, "message": f"Deleted face for '{name}'"}), 200

        except Exception as e:
            logging.error(f"Error in DELETE /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to delete face", "details": str(e)}), 500


@app.route("/delete_all_faces", methods=["DELETE"])
def delete_all_faces():
    """Delete all known faces."""
    global known_faces

    try:
        # Loop through all known faces and delete their image files
        for face in known_faces:
            if os.path.exists(face["path"]):  # Check if the file exists
                os.remove(face["path"])  # Delete the image file

        # Clear the known_faces list
        known_faces = []

        return jsonify({"data": None, "code": 200, "message": "All known faces have been deleted."}), 200

    except Exception as e:
        logging.error(f"Error in delete_all_faces: {e}")
        return jsonify({"data": None, "code": 500, "message": "Failed to delete all faces", "details": str(e)}), 500



if __name__ == "__main__":
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    load_known_faces()
    app.run(host="0.0.0.0", port=8000, debug=True)
