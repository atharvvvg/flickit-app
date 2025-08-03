"""
Enhanced helper functions for better ball detection in dynamic scenarios
"""

import cv2
import numpy as np
from typing import List, Tuple

# detection constants
SPORTS_BALL_CLASS_NAME = 'sports ball'
PERSON_CLASS_NAME = 'person'

CONFIDENCE_THRESHOLD = 0.05
PERSON_CONFIDENCE_THRESHOLD = 0.4

MIN_BALL_SIZE = 2
MAX_BALL_SIZE = 300
MIN_PERSON_SIZE = 50

MOTION_BLUR_CONFIDENCE_REDUCTION = 0.05

def detect_objects(frame: np.ndarray, model) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Enhanced object detection with multi-scale processing and confidence adjustment
    """
    ball_detections = []
    person_detections = []
    
    balls_high_conf, persons_high_conf = _detect_with_confidence(frame, model, CONFIDENCE_THRESHOLD, PERSON_CONFIDENCE_THRESHOLD)
    ball_detections.extend(balls_high_conf)
    person_detections.extend(persons_high_conf)
    
    balls_low_conf, _ = _detect_with_confidence(frame, model, CONFIDENCE_THRESHOLD - MOTION_BLUR_CONFIDENCE_REDUCTION, PERSON_CONFIDENCE_THRESHOLD)
    
    for low_ball in balls_low_conf:
        is_duplicate = False
        for high_ball in ball_detections:
            if calculate_iou(low_ball, high_ball) > 0.3: 
                is_duplicate = True
                break
        if not is_duplicate:
            ball_detections.append(low_ball)
    
    return ball_detections, person_detections

def _detect_with_confidence(frame, model, ball_conf_thresh, person_conf_thresh):
    """Core detection function with specified confidence thresholds"""
    results = model.predict(frame, verbose=False)
    ball_detections = []
    person_detections = []
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            class_names = model.names
            
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                if (class_id < len(class_names) and 
                    class_names[class_id] == SPORTS_BALL_CLASS_NAME and 
                    confidence > ball_conf_thresh):
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if (width >= MIN_BALL_SIZE and height >= MIN_BALL_SIZE and
                        width <= MAX_BALL_SIZE and height <= MAX_BALL_SIZE):
                        
                        aspect_ratio = width / height if height > 0 else 0
                        if 0.3 <= aspect_ratio <= 3.0:
                            ball_detections.append([x1, y1, x2, y2])
                
                # Person detection
                elif (class_id < len(class_names) and 
                      class_names[class_id] == PERSON_CLASS_NAME and 
                      confidence > person_conf_thresh):
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width >= MIN_PERSON_SIZE and height >= MIN_PERSON_SIZE:
                        person_detections.append([x1, y1, x2, y2])
    
    return ball_detections, person_detections

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_proximity(ball_center: Tuple[float, float], person_bbox: List[int]) -> float:
    """Calculate distance between ball center and person bounding box"""
    ball_x, ball_y = ball_center
    px1, py1, px2, py2 = person_bbox
    
    if px1 <= ball_x <= px2 and py1 <= ball_y <= py2:
        return 0.0
    
    closest_x = max(px1, min(ball_x, px2))
    closest_y = max(py1, min(ball_y, py2))
    
    distance = np.sqrt((ball_x - closest_x)**2 + (ball_y - closest_y)**2)
    return distance

def calculate_velocity(track_history: List[Tuple[float, float]], fps: float = 30.0) -> float:
    """Calculate smoothed velocity with outlier rejection"""
    if len(track_history) < 2:
        return 0.0
    
    velocities = []
    window = min(5, len(track_history))
    recent_history = track_history[-window:]
    
    for i in range(1, len(recent_history)):
        x1, y1 = recent_history[i-1]
        x2, y2 = recent_history[i]
        displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        velocity = displacement * fps
        velocities.append(velocity)
    
    if not velocities:
        return 0.0
    
    if len(velocities) > 2:
        velocities = np.array(velocities)
        median_vel = np.median(velocities)
        mad = np.median(np.abs(velocities - median_vel))
        if mad > 0:
            mask = np.abs(velocities - median_vel) <= 3 * mad
            velocities = velocities[mask]
    
    return np.mean(velocities)

def detect_motion_blur(frame_roi):
    """Detect if a region has motion blur (simple Laplacian variance)"""
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY) if len(frame_roi.shape) == 3 else frame_roi
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100 