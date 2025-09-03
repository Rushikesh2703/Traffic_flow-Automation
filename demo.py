from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 Nano model (Fast & Lightweight)

# Vehicle classes to detect
vehicle_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']

# Dictionary to store vehicle counts for each lane
lane_vehicle_counts = {}

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

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Perform detection
        results = model(frame)
        result_frame = results[0].plot()  # Plot detection results on the frame
        
        # Counting Vehicles
        vehicle_count = 0
        for result in results:
            for box in result.boxes:
                cls = model.names[int(box.cls)]  # Class name
                if cls in vehicle_classes:
                    vehicle_count += 1
        
        # Display vehicle count and signal time on the frame
        signal_time = calculate_signal_time(vehicle_count)
        cv2.putText(result_frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Signal Time: {signal_time}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the result frame
        cv2.imshow("Real-Time Traffic Detection", result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
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