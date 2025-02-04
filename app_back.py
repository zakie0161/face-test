from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
import uuid
import logging
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Text, select, delete, update
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import json
from urllib.parse import quote

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Constants
IMAGE_FOLDER = "images"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Get individual database configurations from .env
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_database")
DB_USER = os.getenv("DB_USER", "username")
DB_PASSWORD = quote(os.getenv("DB_PASSWORD", "password"))

if not (DB_HOST and DB_PORT and DB_NAME and DB_USER and DB_PASSWORD):
    raise ValueError("Database configuration is incomplete in the .env file")

# Construct the DATABASE_URL
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Database configuration
engine = create_engine(DB_URL)
metadata = MetaData()
Session = sessionmaker(bind=engine)
session = Session()

# Define table schema
employee_table = Table(
    'employee_face_recognition',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('employee_guid', String, nullable=True),
    Column('employee_id_ref', Integer, nullable=True),
    Column('employee_name', String, nullable=True),
    Column('encoding', Text, nullable=True),  # Storing encoding as a JSON-like string
    Column('images_path', Text, nullable=True)
)

# Helper functions
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encoding_to_string(encoding):
    """Convert a NumPy array to a JSON string."""
    return json.dumps(encoding.tolist())

def string_to_encoding(encoding_str):
    """Convert a JSON string back to a NumPy array."""
    return np.array(json.loads(encoding_str))

def load_known_faces():
    """Load known faces from the database."""
    query = select([employee_table])
    result = session.execute(query).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["employee_name"],
            "encoding": string_to_encoding(row["encoding"]),
            "path": row["images_path"],
        }
        for row in result
    ]

@app.route("/recognize", methods=["POST"])
def recognize_faces():
    """Recognize faces in an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"data": None, "code": 400, "message": "No image file provided"}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({"data": None, "code": 400, "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed."}), 400

    employee_guid_filter = request.form.get('employee_guid', None)

    if not employee_guid_filter:
        return jsonify({"data": None, "code": 400, "message": "No employee_guid provided"}), 400

    try:
        # Read and decode the image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"data": None, "code": 400, "message": "Unable to decode image file"}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if not face_encodings:
            return jsonify({"data": None, "code": 404, "message": "No face detected"}), 404

        # Query the database for faces with the specific employee_guid
        query = select([employee_table])
        if employee_guid_filter:
            query = query.where(employee_table.c.employee_guid == employee_guid_filter)

        known_faces = session.execute(query).fetchall()
        if not known_faces:
            return jsonify({"data": None, "code": 404, "message": "No known faces for the provided employee_guid"}), 404

        # Prepare known faces for comparison
        known_encodings = [string_to_encoding(face["encoding"]) for face in known_faces]
        known_face_info = [
            {
                "id": face["id"],
                "employee_guid": face["employee_guid"],
                "employee_name": face["employee_name"],
                "path": face["images_path"],
            }
            for face in known_faces
        ]

        best_match = None
        highest_confidence = 0

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]

                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = {
                        "id": known_face_info[best_match_index]["id"],
                        "employee_guid": known_face_info[best_match_index]["employee_guid"],
                        "employee_name": known_face_info[best_match_index]["employee_name"],
                        "path": known_face_info[best_match_index]["path"],
                        "confidence": round(confidence, 2),
                    }

        if best_match:
            return jsonify({"data": best_match, "code": 200, "message": "Success"})
        else:
            return jsonify({"data": None, "code": 404, "message": "No matching faces found"}), 404

    except Exception as e:
        logging.error(f"Error in recognize_faces: {e}")
        return jsonify({"data": None, "code": 500, "error": "Failed to process image", "details": str(e)}), 500


@app.route("/faces", methods=["GET", "POST", "PUT", "DELETE"])
def manage_faces():
    """CRUD operations for known faces."""
    global session

    if request.method == "GET":
        # Retrieve all faces from the database
        try:
            query = select([employee_table])
            result = session.execute(query).fetchall()
            faces = [
                {
                    "guid": row["employee_guid"],
                    "name": row["employee_name"],
                    "path": row["images_path"]
                } for row in result
            ]
            return jsonify({"data": faces, "code": 200, "message": "Success"})
        except Exception as e:
            logging.error(f"Error in GET /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to fetch faces", "details": str(e)}), 500

    elif request.method == "POST":
        # Add a new face to the database
        if 'image' not in request.files or 'name' not in request.form or 'employee_guid' not in request.form:
            return jsonify({"data": None, "code": 400, "message": "Image file, employee_guid and name are required"}), 400

        employee_guid = request.form['employee_guid']
        file = request.files['image']
        name = request.form['name']

        if not allowed_file(file.filename):
            return jsonify({"data": None, "code": 400, "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed."}), 400

        try:
            file_extension = file.filename.rsplit('.', 1)[-1].lower()
            random_filename = f"{uuid.uuid4()}.{file_extension}"
            save_path = os.path.join(IMAGE_FOLDER, random_filename)
            file.save(save_path)

            img = cv2.imread(save_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)

            if not encodings:
                os.remove(save_path)
                return jsonify({"data": None, "code": 404, "message": "No face detected in the image"}), 404

            # Insert into database
            insert_query = employee_table.insert().values(
                employee_guid=employee_guid,
                employee_name=name,
                encoding=encoding_to_string(encodings[0]),
                images_path=save_path
            )
            session.execute(insert_query)
            session.commit()

            return jsonify({"data": None, "code": 200, "message": f"Added {name} to known faces", "path": save_path}), 201

        except Exception as e:
            logging.error(f"Error in POST /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to add face", "details": str(e)}), 500

    elif request.method == "PUT":
        # Update an existing face
        if 'employee_guid' not in request.form:
            return jsonify({"data": None, "code": 400, "message": "employee_guid is required to update a face"}), 400

        face_id = request.form['employee_guid']
        new_name = request.form.get('name')  # Optional
        new_image = request.files.get('image')  # Optional

        try:
            query = select([employee_table]).where(employee_table.c.employee_guid == face_id)
            face_record = session.execute(query).fetchone()

            if not face_record:
                return jsonify({"data": None, "code": 404, "message": "Face not found"}), 404

            updates = {}
            if new_name:
                updates["employee_name"] = new_name

            if new_image:
                # Save new image
                file_extension = new_image.filename.rsplit('.', 1)[-1].lower()
                random_filename = f"{uuid.uuid4()}.{file_extension}"
                new_save_path = os.path.join(IMAGE_FOLDER, random_filename)
                new_image.save(new_save_path)

                # Process new image
                img = cv2.imread(new_save_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img_rgb)

                if not encodings:
                    os.remove(new_save_path)
                    return jsonify({"data": None, "code": 404, "message": "No face detected in the new image"}), 404

                # Remove old image
                if os.path.exists(face_record["images_path"]):
                    os.remove(face_record["images_path"])

                updates["encoding"] = encoding_to_string(encodings[0])
                updates["images_path"] = new_save_path

            # Update the database
            update_query = update(employee_table).where(employee_table.c.employee_guid == face_id).values(**updates)
            session.execute(update_query)
            session.commit()

            return jsonify({"data": face_id, "code": 200, "message": "Face updated successfully"}), 200

        except Exception as e:
            logging.error(f"Error in PUT /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to update face", "details": str(e)}), 500

    elif request.method == "DELETE":
        # Delete a face by ID
        face_id = request.args.get('employee_guid')
        if not face_id:
            return jsonify({"data": None, "code": 400, "message": "employee_guid is required to delete a face"}), 400

        try:
            query = select([employee_table]).where(employee_table.c.employee_guid == face_id)
            face_record = session.execute(query).fetchone()

            if not face_record:
                return jsonify({"data": None, "code": 404, "message": "Face not found"}), 404

            # Delete from database
            delete_query = delete(employee_table).where(employee_table.c.employee_guid == face_id)
            session.execute(delete_query)
            session.commit()

                  # Remove image file
            if os.path.exists(face_record["images_path"]):
                os.remove(face_record["images_path"])

            return jsonify({"data": face_id, "code": 200, "message": "Face deleted successfully"}), 200

        except Exception as e:
            logging.error(f"Error in DELETE /faces: {e}")
            return jsonify({"data": None, "code": 500, "message": "Failed to delete face", "details": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    metadata.create_all(engine)  # Ensure the table is created
    app.run(host="0.0.0.0", port=8000, debug=True)