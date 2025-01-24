import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Directory to save unique faces
output_dir = "saved_faces"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate histogram for a face region
def calculate_histogram(image, rect):
    x, y, w, h = rect
    face_roi = image[y:y+h, x:x+w]
    hist = cv2.calcHist([face_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to check if a face is unique
def is_unique_face(hist, known_faces, threshold=0.4):
    for known_hist in known_faces:
        similarity = cv2.compareHist(hist, known_hist, cv2.HISTCMP_CORREL)
        if similarity > threshold:
            return False
    return True

# Initialize face mesh
cap = cv2.VideoCapture("video.mp4")

# To store histograms of unique faces
unique_faces_histograms = []

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=100,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    face_counter = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find the face mesh
        results = face_mesh.process(image)

        # Convert back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face mesh landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box of the face
                h, w, _ = image.shape
                xs = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                ys = [int(landmark.y * h) for landmark in face_landmarks.landmark]
                x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
                y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

                # Define face rectangle
                face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Calculate histogram for the face
                hist = calculate_histogram(image, face_rect)

                # Check if the face is unique
                if is_unique_face(hist, unique_faces_histograms):
                    # Save the face if unique
                    unique_faces_histograms.append(hist)
                    face_counter += 1
                    face_image = image[y_min:y_max, x_min:x_max]
                    cv2.imwrite(os.path.join(output_dir, f"face_{face_counter}.jpg"), face_image)

                # Draw the face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Display the output
        cv2.imshow('MediaPipe Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
