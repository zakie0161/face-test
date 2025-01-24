import cv2
import face_recognition
import os
import numpy as np
import dlib

print("CUDA available:", dlib.DLIB_USE_CUDA)


# Folder berisi gambar wajah
face_folder = "detected_faces"  # Ganti dengan folder Anda

# Load semua gambar referensi dari folder dan encode wajahnya
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(face_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        # Load gambar
        image_path = os.path.join(face_folder, file_name)
        image = face_recognition.load_image_file(image_path)
        
        # Encode wajah (asumsi 1 wajah per gambar)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(file_name)[0])  # Nama file sebagai nama wajah

# Akses kamera menggunakan OpenCV
video_capture = cv2.VideoCapture("rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=0")

# Periksa apakah kamera berhasil dibuka
if not video_capture.isOpened():
    print("Gagal membuka kamera")
    exit()

# Gunakan model DNN untuk deteksi wajah (pre-trained model dari Caffe)
face_proto = "deploy.prototxt"  # Path ke file prototxt
face_model = "res10_300x300_ssd_iter_140000.caffemodel"  # Path ke file model

# Load model DNN dengan CUDA
net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Set CUDA backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Set target CUDA

print("Tekan 'q' untuk keluar")

while True:
    # Tangkap frame dari kamera
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break

    # Mengirim frame ke GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Konversi frame ke format yang diterima oleh DNN
    blob = cv2.dnn.blobFromImage(gpu_frame.download(), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Proses deteksi wajah
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Ambang batas kepercayaan
            # Lokasi wajah
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            # Gambarkan kotak di sekitar wajah
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Tambahkan label nama (menggunakan face recognition)
            face_encoding = face_recognition.face_encodings(frame, [(y1, x2, y2, x1)])
            face = frame[y1:y2, x1:x2]
            if face_encoding:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
                name = "Unknown"
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                # else:   
                #     face_filename = os.path.join("detected_faces", f"face_test.jpg")
                #     cv2.imwrite(face_filename, face)  
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Tampilkan frame dengan kotak dan nama
    cv2.imshow('Video', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
video_capture.release()
cv2.destroyAllWindows()
