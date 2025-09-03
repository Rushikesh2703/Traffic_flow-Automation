import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
vehicle_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']

def get_congestion_level(count):
    if count < 10:
        return 'low'
    elif count < 25:
        return 'medium'
    else:
        return 'high'

def process_frame(frame):
    results = model(frame)
    result_frame = results[0].plot()

    vehicle_count = sum(1 for result in results for box in result.boxes 
                        if model.names[int(box.cls)] in vehicle_classes)

    congestion = get_congestion_level(vehicle_count)

    return result_frame, vehicle_count, congestion

def main():
    st.title("🚦 Adaptive Traffic Signal Controller")

    input_mode = st.radio("Select Input Mode", ["Upload Video", "Use Webcam"])

    if input_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name

            st.video(tfile_path)
            cap = cv2.VideoCapture(tfile_path)

            process_and_display(cap)

    elif input_mode == "Use Webcam":
        run_webcam = st.checkbox("Start Webcam Stream")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            process_and_display(cap)

def process_and_display(cap):
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, vehicle_count, congestion = process_frame(frame)

        # Resize for Streamlit and display
        result_frame = cv2.resize(result_frame, (640, 480))
        stframe.image(result_frame, channels="BGR", caption=f"Vehicles: {vehicle_count}, Congestion: {congestion}")

        # Simulate processing delay
        time.sleep(0.05)

    cap.release()

if __name__ == "__main__":
    main()
