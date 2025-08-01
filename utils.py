"""
Helper functions (e.g., video processing, drawing)
"""

import cv2
import numpy as np
from typing import List

# Constants
SPORTS_BALL_CLASS_NAME = 'sports ball'
CONFIDENCE_THRESHOLD = 0.4

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
                    
                    detections.append([x1, y1, x2, y2])
    
    return detections