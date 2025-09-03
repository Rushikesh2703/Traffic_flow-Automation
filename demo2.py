from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import queue

# Load pre-trained YOLOv8 model (nano for speed)
model = YOLO('yolov8n.pt')  # Use GPU and half-precision

# Vehicle classes to detect
vehicle_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']

def calculate_signal_time(vehicle_count):
    if vehicle_count <= 5:
        time = 10  # Minimum signal time for <= 5 vehicles
    elif 6 <= vehicle_count <= 25:
        time = 10 + (vehicle_count - 5) * 2  # Linear adjustment between 10 and 50
    elif vehicle_count > 25:
        time = 50 + (vehicle_count - 25) * 0.8  # Increase from 50 up to max 70
    else:
        time = 50  # Default signal time for 25 vehicles
    
    return max(10, min(time, 70))  # Ensure time is between 10 and 70 seconds

class VideoCaptureThread:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        self.queue = queue.Queue(maxsize=1)
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(None)  # Signal end of video
                break
            if not self.queue.full():
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def release(self):
        self.cap.release()

def process_video(video_source):
    cap = VideoCaptureThread(video_source)
    frame_count = 0
    skip_frames = 2  # Process every 2nd frame

    while True:
        frame = cap.read()
        if frame is None:  # End of video
            print("Video ended.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip this frame

        # Perform detection
        results = model(frame, imgsz=320)  # Reduce input size
        result_frame = results[0].plot()  # Plot detection results

        # Counting Vehicles
        vehicle_count = sum(1 for result in results for box in result.boxes if model.names[int(box.cls)] in vehicle_classes)
        signal_time = calculate_signal_time(vehicle_count)

        # Display vehicle count and signal time
        cv2.putText(result_frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Signal Time: {signal_time}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize for display
        display_frame = cv2.resize(result_frame, (800, 600))
        cv2.imshow("Real-Time Traffic Detection", display_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_real_time_processing():
    video_source = 0  # Use 0 for webcam or provide a video file path
    process_video(video_source)

def upload_and_process_video():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        print("No video file selected")
        return
    process_video(file_path)

# Create GUI window
root = tk.Tk()
root.title("Traffic Vehicle Detection")
root.geometry("400x200")

# Button for real-time webcam processing
btn_webcam = tk.Button(root, text="Start Real-Time Webcam Processing", command=start_real_time_processing)
btn_webcam.pack(pady=10)

# Button for uploading and processing a video file
btn_upload = tk.Button(root, text="Upload and Process Video File", command=upload_and_process_video)
btn_upload.pack(pady=10)

root.mainloop()