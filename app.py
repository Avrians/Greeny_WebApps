from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from ultralytics.solutions import object_counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# Load YOLO model and initialize variables
model = YOLO("model/best40.pt")
region_of_interest = [(20, 600), (1700, 604), (1700, 560), (20, 560)]
counter = object_counter.ObjectCounter()
counter.set_args(reg_pts=region_of_interest, classes_names=model.names, draw_tracks=True)

# Function to capture video from webcam and perform object detection
def webcam_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection and counting
        tracks = model.track(frame, persist=True, show=False)
        frame_with_detection = counter.start_counting(frame, tracks)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_detection)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Function to process uploaded video and perform object detection
def process_uploaded_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open uploaded video.")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection and counting
        tracks = model.track(frame, persist=True, show=False)
        frame_with_detection = counter.start_counting(frame, tracks)

        # Write the frame with detections to the output video
        out.write(frame_with_detection)

    cap.release()
    out.release()
    return output_video_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploadvideo')
def uploadvideo():
    return render_template('detection_upload.html')

@app.route('/detectionrealtime')
def detectionrealtime():
    return render_template('detection_realtime.html')

# Route for streaming video with object detection using webcam
@app.route('/video_feed_webcam')
def video_feed_webcam():
    return Response(webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle video upload form submission
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(request.url)

    # Save the uploaded video to a temporary location
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    # Process the uploaded video and perform object detection
    output_video_path = process_uploaded_video(video_path)

    return redirect(url_for('detection_result', filename='output.mp4'))

@app.route('/detection_result/<filename>')
def detection_result(filename):
    return render_template('detection_result.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
