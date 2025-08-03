"""
Helper functions (e.g., video processing, drawing)
"""

import cv2
import numpy as np
from typing import List, Tuple

# Constants
SPORTS_BALL_CLASS_NAME = 'sports ball'
CONFIDENCE_THRESHOLD = 0.25  # Further lowered to catch more balls
MIN_BALL_SIZE = 12  # Further reduced minimum size
MAX_BALL_SIZE = 200  # Maximum size to avoid false positives

def detect_balls(frame: np.ndarray, model) -> List[List[int]]:
    """
    Detect footballs in a single frame using YOLO model.
    
    Args:
        frame: Input frame as numpy array
        model: YOLO model instance
    
    Returns:
        List of bounding boxes [x1, y1, x2, y2] for detected balls
    """
    # Run YOLO prediction on the frame
    results = model.predict(frame, verbose=False)
    
    # Extract detections
    detections = []
    
    if results and len(results) > 0:
        result = results[0]  # Get first (and only) result
        
        # Check if we have any detections
        if result.boxes is not None:
            boxes = result.boxes
            
            # Get class names from the model
            class_names = model.names
            
            # Filter for sports ball class with confidence > threshold
            for i in range(len(boxes)):
                # Get class index and confidence
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                
                # Check if it's a sports ball with sufficient confidence
                if (class_id < len(class_names) and 
                    class_names[class_id] == SPORTS_BALL_CLASS_NAME and 
                    confidence > CONFIDENCE_THRESHOLD):
                    
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Filter by size
                    width = x2 - x1
                    height = y2 - y1
                    
                    if (width >= MIN_BALL_SIZE and height >= MIN_BALL_SIZE and
                        width <= MAX_BALL_SIZE and height <= MAX_BALL_SIZE):
                        
                        # Additional filtering: check aspect ratio
                        aspect_ratio = width / height if height > 0 else 0
                        if 0.5 <= aspect_ratio <= 2.0:  # Balls should be roughly circular
                            detections.append([x1, y1, x2, y2])
    
    return detections


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_velocity(track_history: List[Tuple[float, float]], fps: float = 30.0) -> float:
    """
    Calculate velocity based on track history.
    
    Args:
        track_history: List of (x, y) center points
        fps: Frames per second of the video
    
    Returns:
        Velocity in pixels per second
    """
    if len(track_history) < 2:
        return 0.0
    
    # Calculate displacement between last two points
    x1, y1 = track_history[-2]
    x2, y2 = track_history[-1]
    
    displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Convert to velocity (pixels per second)
    time_diff = 1.0 / fps  # Time between frames
    velocity = displacement / time_diff
    
    return velocity