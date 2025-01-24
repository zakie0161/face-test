import cv2
import mediapipe as mp
import threading

# List of RTSP URLs for the IP cameras
camera_urls = [
    "rtsp://username:password@ip_address1:port/stream",
    # Add more cameras as needed
]

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menangani aliran video dan pengenalan wajah dengan CUDA
def process_camera(camera_url, camera_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Camera {camera_id} cannot be opened.")
        return

    # Gunakan GPU jika tersedia (MediaPipe secara otomatis akan menggunakan GPU jika ada)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.2) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {camera_id} lost connection.")
                break

            # Pindahkan frame ke GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Gunakan CUDA untuk konversi BGR to RGB
            gpu_rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)

            # Unduh frame dari GPU ke CPU setelah konversi
            rgb_frame = gpu_rgb_frame.download()

            # Deteksi wajah
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Gambarkan kotak deteksi wajah
                    mp_drawing.draw_detection(frame, detection)

            # Tampilkan hasil pengenalan wajah
            cv2.imshow(f"Camera {camera_id}", frame)

            # Jika tombol 'q' ditekan, keluar dari video stream untuk kamera ini
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

# Daftar thread untuk menjalankan setiap kamera secara paralel
threads = []

# Membuat dan memulai thread untuk setiap kamera
for idx, url in enumerate(camera_urls):
    thread = threading.Thread(target=process_camera, args=(url, idx + 1))
    threads.append(thread)
    thread.start()

# Menunggu semua thread selesai
for thread in threads:
    thread.join()

cv2.destroyAllWindows()
