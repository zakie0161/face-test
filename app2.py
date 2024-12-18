from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

# Load known face encodings with their names and file paths
known_faces = []  # Stores dicts with "name", "encoding", and "path"


def load_known_faces():
    """Load preloaded faces from a folder or database."""
    people = []

    for person in people:
        try:
            image = face_recognition.load_image_file(person["path"])
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append({
                    "name": person["name"],
                    "encoding": encodings[0],
                    "path": person["path"]
                })
                print(f"Loaded encoding for {person['name']}")
            else:
                print(f"No face found in {person['path']}")
        except Exception as e:
            print(f"Error loading {person['path']}: {str(e)}")


@app.route("/recognize", methods=["POST"])
def recognize_faces():
    """Recognize faces in an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Optional: Get the name from form data (POST data)
    name_filter = request.form.get('name', None)

    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Unable to decode image file"}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if not face_encodings:
            return jsonify({"error": "No face detected"}), 404

        results = []

        # If a name filter is provided, filter the known faces by the provided name
        filtered_known_faces = [f for f in known_faces if f["name"].lower() == name_filter.lower()] if name_filter else known_faces

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                [f["encoding"] for f in filtered_known_faces], face_encoding)
            face_distances = face_recognition.face_distance(
                [f["encoding"] for f in filtered_known_faces], face_encoding)

            name = "Unknown"
            path = None
            confidence = None

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = filtered_known_faces[best_match_index]["name"]
                path = filtered_known_faces[best_match_index]["path"]
                confidence = 1 - face_distances[best_match_index]

            # Add metadata for the recognized face
            results.append({
                "name": name,
                "path": path if path else "No path available",
                "confidence": round(confidence, 2) if confidence else None,
            })

        return jsonify({"faces": results})

    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500


@app.route("/faces", methods=["GET", "POST", "DELETE"])
def manage_faces():
    """CRUD operations for known faces."""
    global known_faces  # Move this to the top of the function

    if request.method == "GET":
        # Return the list of known faces
        return jsonify({"known_faces": [{"name": f["name"], "path": f["path"]} for f in known_faces]})

    elif request.method == "POST":
        # Add a new known face
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({"error": "Image file and name are required"}), 400

        file = request.files['image']
        name = request.form['name']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        try:
            # Ensure the 'images/' folder exists
            folder_path = "images"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Generate a random filename with the original file extension
            file_extension = file.filename.rsplit('.', 1)[-1].lower()  # Get file extension (e.g., jpg, png)
            random_filename = f"{uuid.uuid4()}.{file_extension}"
            save_path = os.path.join(folder_path, random_filename)

            # Save the uploaded image file to the 'images/' folder
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Unable to decode image file"}), 400

            # Save the file to disk
            cv2.imwrite(save_path, img)

            # Process the saved image for face recognition
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)

            if not encodings:
                # Clean up saved file if no face is detected
                os.remove(save_path)
                return jsonify({"error": "No face detected in the image"}), 404

            # Add the face to the known_faces list
            known_faces.append({
                "name": name,
                "encoding": encodings[0],
                "path": save_path
            })

            return jsonify({"message": f"Added {name} to known faces", "path": save_path}), 201

        except Exception as e:
            return jsonify({"error": "Failed to add face", "details": str(e)}), 500

    # PUT: Update an existing face (image or name)
    elif request.method == "PUT":
        if 'name' not in request.form:
            return jsonify({"error": "Name is required to update a face"}), 400

        name = request.form['name']
        new_name = request.form.get('new_name')  # Optional
        new_image = request.files.get('image')  # Optional

        # Find the face by name
        face_record = next((f for f in known_faces if f["name"] == name), None)
        if not face_record:
            return jsonify({"error": f"No face found with name '{name}'"}), 404

        try:
            # Update the name if provided
            if new_name:
                face_record["name"] = new_name

            # Update the image if provided
            if new_image:
                folder_path = "images"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Generate a random filename for the new image
                file_extension = new_image.filename.rsplit('.', 1)[-1].lower()
                random_filename = f"{uuid.uuid4()}.{file_extension}"
                new_save_path = os.path.join(folder_path, random_filename)

                # Save the new image
                file_bytes = new_image.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    return jsonify({"error": "Unable to decode new image file"}), 400

                cv2.imwrite(new_save_path, img)

                # Process the new image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img_rgb)

                if not encodings:
                    os.remove(new_save_path)  # Cleanup if no face detected
                    return jsonify({"error": "No face detected in the new image"}), 404

                # Delete the old image
                old_save_path = face_record["path"]
                if os.path.exists(old_save_path):
                    os.remove(old_save_path)

                # Update the face record
                face_record["encoding"] = encodings[0]
                face_record["path"] = new_save_path

            return jsonify({"message": f"Updated face for '{name}'"}), 200

        except Exception as e:
            return jsonify({"error": "Failed to update face", "details": str(e)}), 500

    # DELETE: Remove a face by name
    elif request.method == "DELETE":
        if 'name' not in request.args:
            return jsonify({"error": "Name is required to delete a face"}), 400

        name = request.args['name']
        face_record = next((f for f in known_faces if f["name"] == name), None)

        if not face_record:
            return jsonify({"error": f"No face found with name '{name}'"}), 404

        try:
            # Remove the image file
            if os.path.exists(face_record["path"]):
                os.remove(face_record["path"])

            # Remove the record from the list
            known_faces.remove(face_record)

            return jsonify({"message": f"Deleted face for '{name}'"}), 200

        except Exception as e:
            return jsonify({"error": "Failed to delete face", "details": str(e)}), 500



if __name__ == "__main__":
    # Create the images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    load_known_faces()
    app.run(host="0.0.0.0", port=8000, debug=True)


# curl -X GET http://localhost:8000/faces
# curl -X DELETE "http://localhost:8000/faces?name=Person 1"
# curl -X POST -F "image=@path/to/new/face.jpg" -F "name=New Person" http://localhost:8000/faces
# curl -X PUT -F "name=John Doe" -F "image=@/path/to/new_image.jpg" http://localhost:8000/faces

