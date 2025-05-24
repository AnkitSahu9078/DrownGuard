import os
import cv2
import numpy as np
import torch
import joblib
import argparse
import cvlib as cv
from cvlib.object_detection import draw_bbox
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from PIL import Image
from flask import Flask, render_template, request, Response, jsonify
import threading
import time
from werkzeug.utils import secure_filename

# Define CNN model architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
processing = False
video_path = None
detection_results = []
current_frame = None
processing_thread = None

# Load model and label binarizer
print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')
model = CustomCNN(len(lb.classes_))
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Model loaded successfully')

# Image transformations
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_drowning():
    global processing, video_path, detection_results, current_frame
    
    # Reset results for new processing
    detection_results = []
    
    if not os.path.exists(video_path):
        processing = False
        return
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print('Error while trying to read video')
        processing = False
        return
    
    # Get video properties for better tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    
    while cap.isOpened() and processing:
        start_time = time.time()
        status, frame = cap.read()
        
        if not status:
            break
            
        frame_count += 1
        
        # Skip frames for faster processing if needed
        if frame_count % 5 != 0:
            continue
            
        # Apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)
        
        is_drowning = False
        drowning_probability = 0
        
        # If only one person is detected, use model-based detection
        if len(bbox) == 1:
            bbox0 = bbox[0]
            
            # Process with ML model
            with torch.no_grad():
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                pil_image = pil_image.unsqueeze(0)
                outputs = model(pil_image)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs.data, 1)
            
            prediction_class = lb.classes_[preds]
            drowning_probability = float(probs[0][preds].item())
            
            if prediction_class == 'drowning':
                is_drowning = True
            
        # If more than one person is detected, use logic-based detection
        elif len(bbox) > 1:
            # Calculate the centroid of each bounding box
            centres = []
            for i in range(len(bbox)):
                bbox_i = bbox[i]
                centre_i = [(bbox_i[0] + bbox_i[2])/2, (bbox_i[1] + bbox_i[3])/2]
                centres.append(centre_i)
            
            # Calculate the distance between each pair of centroids
            distances = []
            for i in range(len(centres)):
                for j in range(i+1, len(centres)):
                    dist = np.sqrt((centres[i][0] - centres[j][0])**2 + (centres[i][1] - centres[j][1])**2)
                    distances.append(dist)
            
            # If the minimum distance is less than a threshold, consider it as drowning
            if len(distances) > 0 and min(distances) < 50:
                is_drowning = True
                drowning_probability = 0.85  # Estimated probability based on distance
        
        # Draw bounding boxes
        out_frame = draw_bbox(frame, bbox, label, conf, is_drowning)
        
        # Add text for drowning status
        status_text = "DROWNING DETECTED!" if is_drowning else "Normal"
        color = (0, 0, 255) if is_drowning else (0, 255, 0)
        cv2.putText(out_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add probability
        prob_text = f"Confidence: {drowning_probability:.2f}"
        cv2.putText(out_frame, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate processing time and progress
        processing_time = time.time() - start_time
        progress = min(100, (frame_count / total_frames) * 100) if total_frames > 0 else 0
        
        # Store detection result with more info
        detection_results.append({
            'frame': frame_count,
            'drowning': is_drowning,
            'probability': drowning_probability,
            'num_people': len(bbox),
            'processing_time': processing_time * 1000,  # Convert to ms
            'progress': progress
        })
        
        # Convert to JPEG for web display
        _, buffer = cv2.imencode('.jpg', out_frame)
        current_frame = buffer.tobytes()
        
        # Simulate real-time processing
        time.sleep(0.05)
    
    cap.release()
    processing = False
    print(f"Processing completed - analyzed {frame_count} frames")

def gen_frames():
    global current_frame
    while processing:
        if current_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global processing, video_path, processing_thread
    
    if processing:
        return jsonify({'status': 'error', 'message': 'Already processing a video'})
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'})
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Start processing in background thread
        video_path = file_path
        processing = True
        processing_thread = threading.Thread(target=detect_drowning)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Processing started'})
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'})

@app.route('/video_feed')
def video_feed():
    if processing:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(b'', mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global detection_results
    
    # Calculate summary statistics if we have results
    stats = {}
    if detection_results:
        drowning_alerts = sum(1 for r in detection_results if r['drowning'])
        avg_confidence = sum(r['probability'] for r in detection_results) / len(detection_results)
        avg_processing = sum(r['processing_time'] for r in detection_results) / len(detection_results)
        
        stats = {
            'total_frames': len(detection_results),
            'drowning_alerts': drowning_alerts,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing,
            'progress': detection_results[-1]['progress'] if detection_results else 0
        }
    
    return jsonify({
        'processing': processing,
        'results': detection_results,  # Return all results
        'stats': stats
    })

@app.route('/stop')
def stop_processing():
    global processing
    processing = False
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=1.0)
    return jsonify({'status': 'success', 'message': 'Processing stopped'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)