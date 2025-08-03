# Football Tracking System

This system detects, tracks, and classifies footballs in video footage, distinguishing between an "action ball" (the ball being actively used) and "stationary balls" (balls at rest).

## Features

- **Multi-object tracking** using Kalman Filter + Hungarian Algorithm
- **Automatic ball classification** based on motion analysis
- **Professional visualization** with proper annotations
- **Robust detection** with improved sensitivity

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Place your input video file in the `input/` directory as `test.mp4`

## Usage

Run the main script:

```bash
python main.py
```

The processed video will be saved as `output/output1.mp4`.

## System Architecture

### Detection (utils.py)

- Uses YOLOv8 for ball detection
- Filters for sports ball class with confidence > 0.25
- Applies size and aspect ratio filtering

### Tracking (tracker.py)

- Kalman Filter for state prediction
- Hungarian Algorithm for optimal data association
- Robust track lifecycle management

### Visualization (main.py)

- Action ball: Cyan circle with fading trail
- Stationary balls: Green bounding boxes with red center dots
- Professional annotations: "ID X | STATUS | V=velocity"

## Output Format

The system produces videos with:

- **Action Ball**: Central circle with fading trail showing movement history
- **Stationary Balls**: Bounding boxes with unique IDs
- **Annotations**: Professional text labels with ID, status, and velocity

## Configuration

Key parameters can be adjusted in the respective files:

- `CONFIDENCE_THRESHOLD` in utils.py (detection sensitivity)
- `iou_threshold` in tracker.py (tracking robustness)
- `max_age` in tracker.py (track persistence)
- `classification_frame` in tracker.py (when to classify balls)

## Troubleshooting

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install ultralytics opencv-python numpy scipy
```

For better detection results, try adjusting the confidence threshold in `utils.py` if balls are being missed.
