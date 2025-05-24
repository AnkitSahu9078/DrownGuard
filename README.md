# DrownGuard: ML-Powered Drowning Detection and Alert Interface

DrownGuard is an advanced drowning detection system that uses machine learning and computer vision to identify potential drowning incidents and trigger alerts in real-time.

## Features

- Real-time drowning detection using computer vision
- Web-based interface for monitoring
- Alerts when potential drowning is detected
- Supports both video file input and webcam streams

## Technologies Used

- Python
- OpenCV
- PyTorch
- Flask
- YOLO (You Only Look Once) object detection
- Bootstrap

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DrownGuard.git
   cd DrownGuard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required model files:
   - The application will automatically download YOLO weights and configuration files on first run

## Usage

### Web Interface

1. Start the Flask web server:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Upload a video file or use a webcam stream to begin drowning detection

### Command Line Interface

For direct video file analysis:
```bash
python DrownDetect.py --source your_video_file.mp4
```

For webcam input:
```bash
python DrownDetect.py --source 0
```

## How It Works

DrownGuard uses a two-stage detection approach:

1. **Person Detection**: Uses YOLO to identify people in the video frame  
2. **Drowning Classification**:  
   - For single person: Uses a custom CNN model to classify swimming behavior as normal or drowning  
   - For multiple people: Uses proximity-based logic to detect potential drowning incidents

When drowning is detected, the system:
- Displays a visual alert with red bounding boxes
- Triggers an audio alarm

## Project Structure

- `app.py`: Flask web application  
- `DrownDetect.py`: Command-line interface for drowning detection  
- `cvlib/`: Custom computer vision library  
- `templates/`: HTML templates for the web interface  
- `static/`: CSS, JavaScript, and other static assets  
- `model.pth`: PyTorch model for drowning detection  
- `lb.pkl`: Label binarizer for classification  

## License

[MIT License](LICENSE)

## Acknowledgments

- YOLO for object detection  
- PyTorch for deep learning framework