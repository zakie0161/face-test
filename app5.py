import cv2
import numpy as np

# Load the video
video_path = "rtsp://admin:@WARMUP123@4ca904bdbbde.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(video_path)

# Desired output dimensions (16:9 aspect ratio)
desired_width = 1280
desired_height = int(desired_width * 9 / 16)

# Initialize VideoWriter for saving the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_16_9.mp4", fourcc, fps, (desired_width, desired_height))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break when the video ends

    # Get original frame dimensions
    (h, w) = frame.shape[:2]

    # Calculate scaling factor to maintain original aspect ratio
    scale = min(desired_width / w, desired_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize frame while keeping aspect ratio
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Add black padding to make it 16:9
    canvas = np.zeros((desired_height, desired_width, 3), dtype="uint8")  # Black canvas
    x_offset = (desired_width - new_width) // 2
    y_offset = (desired_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    # Write the resized frame to the output video
    out.write(canvas)

    # (Optional) Show the resized frame for debugging
    cv2.imshow("16:9 Frame", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
