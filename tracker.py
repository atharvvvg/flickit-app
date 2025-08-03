# Contains the core multi-object tracking logic

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from utils import calculate_iou

class Tracker:
    """Multi-object tracker for football detection"""
    
    def __init__(self):
        """Initialize the tracker"""
        self.tracks = []
        self.next_track_id = 0
        # Track lifecycle parameters
        self.max_age = 45  # Increased maximum frames to keep unmatched tracks
        self.min_hits = 1  # Minimum hits before track is confirmed
        self.iou_threshold = 0.1  # Further reduced IoU threshold for better matching
        
        # Kalman Filter parameters
        self.process_noise = 0.02  # Reduced process noise for smoother tracking
        self.measurement_noise = 0.08  # Reduced measurement noise
        
        # Ball classification parameters
        self.frame_count = 0  # Current frame count
        self.classification_frame = 45  # Frame at which to perform classification
        self.action_ball_id = None  # ID of the action ball
        self.classification_done = False  # Whether classification has been performed
        
        # Track velocity calculation
        self.fps = 30.0  # Default FPS, will be updated from video
        
        # Track management
        self.min_track_length = 5  # Minimum track length before considering for classification
    
    def set_fps(self, fps: float):
        """Set the FPS for velocity calculations"""
        self.fps = fps
    
    def _create_kalman_filter(self):
        """Create a Kalman Filter for tracking"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        
        # State transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], np.float32)
        
        # Measurement matrix (we only measure position, not velocity)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * self.process_noise
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * self.measurement_noise
        
        return kalman
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection bounding boxes [x1, y1, x2, y2]
        
        Returns:
            List of active tracks
        """
        # Increment frame count
        self.frame_count += 1
        
        # If no existing tracks, create new tracks for all detections
        if len(self.tracks) == 0:
            for detection in detections:
                self._create_new_track(detection)
            return self.tracks
        
        # Predict new positions for all tracks using Kalman Filter
        for track in self.tracks:
            prediction = track['kalman'].predict()
            # Update predicted bbox based on predicted center
            x1, y1, x2, y2 = track['bbox']
            old_center_x = (x1 + x2) / 2
            old_center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            new_center_x = prediction[0][0]
            new_center_y = prediction[1][0]
            
            new_x1 = int(new_center_x - width / 2)
            new_y1 = int(new_center_y - height / 2)
            new_x2 = int(new_center_x + width / 2)
            new_y2 = int(new_center_y + height / 2)
            
            track['predicted_bbox'] = [new_x1, new_y1, new_x2, new_y2]
        
        # If no detections, increment age of all tracks
        if len(detections) == 0:
            for track in self.tracks:
                track['age'] += 1
                track['time_since_update'] += 1
            # Remove old tracks
            self.tracks = [track for track in self.tracks if track['age'] < self.max_age]
            return self.tracks
        
        # Create cost matrix for Hungarian algorithm using predicted positions
        # Cost = 1 - IoU (lower cost = better match)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # Use predicted bbox for matching
                predicted_bbox = track['predicted_bbox']
                iou = calculate_iou(predicted_bbox, detection)
                cost_matrix[i, j] = 1 - iou  # Convert IoU to cost
        
        # Use Hungarian algorithm to find optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Process matches
        matched_tracks = set()
        matched_detections = set()
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            # Only accept matches with IoU > threshold
            if cost_matrix[track_idx, detection_idx] < (1 - self.iou_threshold):
                # Update track with new detection
                self.tracks[track_idx]['bbox'] = detections[detection_idx]
                self.tracks[track_idx]['age'] = 0
                self.tracks[track_idx]['hits'] += 1
                self.tracks[track_idx]['time_since_update'] = 0
                
                # Update Kalman Filter with new measurement
                x1, y1, x2, y2 = detections[detection_idx]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                measurement = np.array([[center_x], [center_y]], np.float32)
                self.tracks[track_idx]['kalman'].correct(measurement)
                
                # Update center history
                self.tracks[track_idx]['center_history'].append((center_x, center_y))
                
                matched_tracks.add(track_idx)
                matched_detections.add(detection_idx)
        
        # Handle unmatched tracks (increment age and use predicted position)
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['age'] += 1
                track['time_since_update'] += 1
                # Use predicted position for unmatched tracks
                track['bbox'] = track['predicted_bbox']
                
                # Add predicted center to history for unmatched tracks
                x1, y1, x2, y2 = track['predicted_bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                track['center_history'].append((center_x, center_y))
        
        # Handle unmatched detections (create new tracks)
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                # Check if this detection is close to any existing unmatched track
                should_create_new = True
                for i, track in enumerate(self.tracks):
                    if i not in matched_tracks:
                        iou = calculate_iou(track['bbox'], detection)
                        if iou > 0.05:  # Reduced threshold for better track creation
                            should_create_new = False
                            break
                
                if should_create_new:
                    self._create_new_track(detection)
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track['age'] < self.max_age]
        
        # Record center points for classification (first 45 frames only)
        if self.frame_count <= self.classification_frame and not self.classification_done:
            for track in self.tracks:
                if len(track['center_history']) < self.frame_count:
                    x1, y1, x2, y2 = track['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    track['center_history'].append((center_x, center_y))
        
        # Perform classification at frame 45
        if self.frame_count == self.classification_frame and not self.classification_done:
            self.classify_balls()
        
        return self.tracks
    
    def _create_new_track(self, detection):
        """Create a new track for a detection"""
        x1, y1, x2, y2 = detection
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        kalman = self._create_kalman_filter()
        kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        track = {
            'id': self.next_track_id,
            'bbox': detection,
            'age': 0,
            'hits': 1,
            'time_since_update': 0,
            'kalman': kalman,
            'center_history': [(center_x, center_y)]
        }
        self.tracks.append(track)
        self.next_track_id += 1
    
    def get_track_stats(self):
        """Get statistics about current tracks for debugging"""
        if not self.tracks:
            return "No active tracks"
        
        stats = []
        for track in self.tracks:
            stats.append(f"ID: {track['id']}, Age: {track['age']}, Hits: {track['hits']}")
        
        return f"Active tracks: {len(self.tracks)} - " + "; ".join(stats)
    
    def classify_balls(self):
        """Classify balls as action ball or stationary balls based on motion"""
        if not self.tracks:
            return
        
        # Calculate standard deviation of positions for each track
        track_motion_scores = []
        
        for track in self.tracks:
            if len(track['center_history']) < self.min_track_length:
                continue  # Skip tracks with insufficient history
            
            # Extract x and y coordinates
            x_coords = [point[0] for point in track['center_history']]
            y_coords = [point[1] for point in track['center_history']]
            
            # Calculate standard deviation of positions
            x_std = np.std(x_coords)
            y_std = np.std(y_coords)
            total_std = np.sqrt(x_std**2 + y_std**2)  # Combined standard deviation
            
            track_motion_scores.append((track['id'], total_std))
        
        if track_motion_scores:
            # Find the track with highest standard deviation (most motion)
            action_track_id = max(track_motion_scores, key=lambda x: x[1])[0]
            self.action_ball_id = action_track_id
            self.classification_done = True
            
            print(f"Action ball identified: Track ID {action_track_id}")
            print(f"Motion scores: {track_motion_scores}")
        else:
            print("No tracks available for classification")
    
    def is_action_ball(self, track_id):
        """Check if a track ID corresponds to the action ball"""
        return self.classification_done and track_id == self.action_ball_id
    
    def get_track_velocity(self, track_id):
        """Get the current velocity of a track"""
        for track in self.tracks:
            if track['id'] == track_id:
                if len(track['center_history']) >= 2:
                    # Calculate velocity from last two points
                    x1, y1 = track['center_history'][-2]
                    x2, y2 = track['center_history'][-1]
                    displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    velocity = displacement * self.fps  # pixels per second
                    return velocity
                else:
                    return 0.0
        return 0.0