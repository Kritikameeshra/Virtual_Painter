# Virtual Painter

A Python application that uses OpenCV and MediaPipe to create a virtual painting experience using hand gestures.

## Features

- Draw on screen using finger movements
- Hand gesture recognition for drawing and erasing
- Real-time webcam feed with drawing overlay

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python virtual_painter.py
   ```

2. Controls:
   - To draw: Hold up your index finger
   - To erase: Hold up both index and middle fingers
   - To quit: Press 'q'

## How it Works

The application uses:
- MediaPipe for hand landmark detection
- OpenCV for image processing and drawing
- Your webcam feed as the input source

The program tracks your hand movements and allows you to draw on the screen by raising specific fingers. The drawing is overlaid on the webcam feed in real-time.
