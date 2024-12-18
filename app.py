from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2

app = Flask(__name__)

# Load known face encodings with their names and file paths
known_face_encodings = []
known_face_names = []
known_face_paths = []

# Preloaded images of known people
def load_known_faces():
    people = [
        {"name": "Person 1", "path": "images/person1.jpg"},
        {"name": "Person 2", "path": "images/person2.jpg"},
        {"name": "Eminem", "path": "images/eminem.jpg"},
    ]

    for person in people:
        try:
            # Load image and extract face encodings
            image = face_recognition.load_image_file(person["path"])
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure at least one encoding is found
                known_face_encodings.append(encodings[0])
                known_face_names.append(person["name"])
                known_face_paths.append(person["path"])
                print(f"Loaded encoding for {person['name']}")
            else:
                print(f"No face found in {person['path']}")
        except Exception as e:
            print(f"Error loading {person['path']}: {str(e)}")

# Load faces when the app starts
load_known_faces()

@app.route("/recognize", methods=["POST"])
def recognize_face():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file in bytes and decode it with OpenCV
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Unable to decode image file"}), 400

        # Convert BGR to RGB as face_recognition expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if not face_encodings:
            return jsonify({"error": "No face detected"}), 404

        # Compare only the first detected face
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        path = None

        if True in matches:
            # Find the index of the first matched face
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            path = known_face_paths[first_match_index]

        # Return only the first result
        return jsonify({
            "name": name,
            "path": path if path else "No path available"
        })

    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
