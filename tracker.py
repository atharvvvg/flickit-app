"""
tracking logic
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from utils import calculate_iou

class Tracker:
    """
    Multi-object tracker for football and person detection with enhanced handling
    for dynamic ball movement, early classification, and robust ID consistency.
    """
    
    def __init__(self):
        """Initialize the tracker"""
        self.ball_tracks = []
        self.person_tracks = []
        self.next_ball_track_id = 0
        self.next_person_track_id = 100
        
        # Tracking Parameters
        self.max_age = 25
        self.action_ball_max_age = 60
        self.min_hits = 1
        self.iou_threshold = 0.15
        
        self.proximity_threshold_tracking = 80
        self.process_noise = 2.5
        self.measurement_noise = 0.3
        
        self.frame_count = 0
        self.warmup_frames = 20
        self.reclassify_interval = 10
        self.action_ball_id = None
        
        # Motion Analysis Parameters
        self.fps = 30.0
        self.min_track_length_for_motion = 3
        self.velocity_smoothing_window = 3
        
        # Ball Classification Thresholds
        self.motion_threshold_base = 15.0
        self.recent_motion_weight = 0.7
        
        # Person Proximity Parameters
        self.proximity_threshold = 120
        self.proximity_boost_factor = 2.0
    
    def set_fps(self, fps: float):
        self.fps = fps if fps > 0 else 30.0
    
    def _create_kalman_filter(self):
        """Create Kalman Filter optimized for dynamic ball motion"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], np.float32)
        
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100
        
        return kalman

    def update(self, ball_detections, person_detections):
        """Update tracker with new detections and perform continuous classification"""
        self.frame_count += 1
        
        self._update_tracks(self.ball_tracks, ball_detections, 'ball')
        self._update_tracks(self.person_tracks, person_detections, 'person')
        
        # Earlier and more frequent classification
        if (self.frame_count >= self.warmup_frames and 
            (self.frame_count - self.warmup_frames) % self.reclassify_interval == 0):
            previous_action_ball = self.action_ball_id
            self.classify_balls()
            if self.action_ball_id != previous_action_ball:
                print(f"[Frame {self.frame_count}] Action ball: ID {previous_action_ball} → {self.action_ball_id}")

        active_ball_tracks = [t for t in self.ball_tracks 
                            if t['hits'] >= self.min_hits and t['time_since_update'] <= 10]
        active_person_tracks = [t for t in self.person_tracks 
                              if t['hits'] >= self.min_hits and t['time_since_update'] <= 10]
        
        return active_ball_tracks, active_person_tracks
    
    def _update_tracks(self, tracks, detections, track_type):
        """Enhanced track update with fallback proximity matching"""
        for track in tracks:
            prediction = track['kalman'].predict()
            w, h = (track['bbox'][2] - track['bbox'][0]), (track['bbox'][3] - track['bbox'][1])
            track['predicted_bbox'] = [
                int(prediction[0, 0] - w / 2), int(prediction[1, 0] - h / 2),
                int(prediction[0, 0] + w / 2), int(prediction[1, 0] + h / 2)
            ]

        matched_indices, unmatched_track_indices, unmatched_detection_indices = \
            self._associate_detections_to_tracks(tracks, detections)

        if track_type == 'ball' and unmatched_track_indices and unmatched_detection_indices:
            proximity_matches = self._proximity_based_association(
                [tracks[i] for i in unmatched_track_indices],
                [detections[i] for i in unmatched_detection_indices]
            )
            
            for track_local_idx, detection_local_idx in proximity_matches:
                track_global_idx = unmatched_track_indices[track_local_idx]
                detection_global_idx = unmatched_detection_indices[detection_local_idx]
                matched_indices.append((track_global_idx, detection_global_idx))
                unmatched_track_indices.remove(track_global_idx)
                unmatched_detection_indices.remove(detection_global_idx)

        for track_idx, detection_idx in matched_indices:
            track = tracks[track_idx]
            detection = detections[detection_idx]
            
            track['bbox'] = detection
            track['time_since_update'] = 0
            track['hits'] += 1
            
            center = ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
            measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
            track['kalman'].correct(measurement)
            track['center_history'].append(center)
            
            if len(track['center_history']) > 100:
                track['center_history'] = track['center_history'][-50:]

        for track_idx in unmatched_track_indices:
            track = tracks[track_idx]
            track['time_since_update'] += 1
            track['bbox'] = track['predicted_bbox']
            
            pred_center = (
                (track['predicted_bbox'][0] + track['predicted_bbox'][2]) / 2,
                (track['predicted_bbox'][1] + track['predicted_bbox'][3]) / 2
            )
            track['center_history'].append(pred_center)

        for detection_idx in unmatched_detection_indices:
            self._create_new_track(detections[detection_idx], track_type)

        self._cleanup_tracks(tracks, track_type)

    def _proximity_based_association(self, unmatched_tracks, unmatched_detections):
        """Fallback association based on proximity for fast-moving objects"""
        if not unmatched_tracks or not unmatched_detections:
            return []
        
        matches = []
        used_detections = set()
        
        for i, track in enumerate(unmatched_tracks):
            best_distance = float('inf')
            best_detection_idx = -1
            
            track_center = (
                (track['predicted_bbox'][0] + track['predicted_bbox'][2]) / 2,
                (track['predicted_bbox'][1] + track['predicted_bbox'][3]) / 2
            )
            
            for j, detection in enumerate(unmatched_detections):
                if j in used_detections:
                    continue
                    
                detection_center = ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
                distance = np.linalg.norm(np.array(track_center) - np.array(detection_center))
                
                if distance < best_distance and distance <= self.proximity_threshold_tracking:
                    best_distance = distance
                    best_detection_idx = j
            
            if best_detection_idx != -1:
                matches.append((i, best_detection_idx))
                used_detections.add(best_detection_idx)
                print(f"Proximity match: Track center {track_center} → Detection {best_detection_idx} (dist: {best_distance:.1f})")
        
        return matches

    def _associate_detections_to_tracks(self, tracks, detections):
        """Associates detections to tracks using Hungarian algorithm with IoU"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou = calculate_iou(track['predicted_bbox'], detection)
                cost_matrix[i, j] = 1 - iou
        
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_detection_indices = set(range(len(detections)))

        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, detection_idx] < (1 - self.iou_threshold):
                matched_indices.append((track_idx, detection_idx))
                unmatched_track_indices.discard(track_idx)
                unmatched_detection_indices.discard(detection_idx)
        
        return matched_indices, list(unmatched_track_indices), list(unmatched_detection_indices)

    def _create_new_track(self, detection, track_type):
        """Creates a new track with enhanced initialization"""
        x1, y1, x2, y2 = detection
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        kalman = self._create_kalman_filter()
        kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        next_id = self.next_ball_track_id if track_type == 'ball' else self.next_person_track_id
        
        track = {
            'id': next_id, 'type': track_type, 'bbox': detection,
            'time_since_update': 0, 'hits': 1, 'kalman': kalman,
            'center_history': [(center_x, center_y)], 'predicted_bbox': detection
        }
        
        if track_type == 'ball':
            self.ball_tracks.append(track)
            self.next_ball_track_id += 1
            print(f"Created new ball track ID {next_id} at ({center_x:.1f}, {center_y:.1f})")
        else:
            self.person_tracks.append(track)
            self.next_person_track_id += 1

    def _cleanup_tracks(self, tracks, track_type):
        """Enhanced track cleanup with action ball protection"""
        tracks_to_keep = []
        
        for track in tracks:
            if track_type == 'ball' and self.is_action_ball(track['id']):
                max_age_for_track = self.action_ball_max_age
            else:
                max_age_for_track = self.max_age
            if track['time_since_update'] < max_age_for_track:
                tracks_to_keep.append(track)
            else:
                print(f"Removing {track_type} track ID {track['id']} (age: {track['time_since_update']})")
        
        tracks[:] = tracks_to_keep

    def classify_balls(self):
        """Enhanced ball classification with recent motion emphasis"""
        if not self.ball_tracks:
            self.action_ball_id = None
            return

        track_scores = []
        
        for track in self.ball_tracks:
            if len(track['center_history']) < self.min_track_length_for_motion:
                continue
            
            total_motion_score = self._calculate_weighted_motion_score(track)
            proximity_boost = self._calculate_proximity_boost(track)
            
            final_score = total_motion_score * proximity_boost
            
            track_scores.append({
                'id': track['id'], 
                'score': final_score,
                'motion': total_motion_score,
                'proximity': proximity_boost
            })
        
        if track_scores:
            best_track = max(track_scores, key=lambda x: x['score'])
            self.action_ball_id = best_track['id']
            print(f"[Frame {self.frame_count}] Ball classification:")
            for score_data in sorted(track_scores, key=lambda x: x['score'], reverse=True):
                status = "ACTION" if score_data['id'] == self.action_ball_id else "STATIONARY"
                print(f"  ID {score_data['id']}: {status} (motion: {score_data['motion']:.1f}, "
                      f"proximity: {score_data['proximity']:.1f}, final: {score_data['score']:.1f})")
        else:
            self.action_ball_id = None

    def _calculate_weighted_motion_score(self, track):
        """Calculate motion score with emphasis on recent movement"""
        history = track['center_history']
        
        if len(history) < 3:
            return 0.0
        
        # Historical motion (standard deviation of all positions)
        x_coords = [p[0] for p in history]
        y_coords = [p[1] for p in history]
        historical_motion = np.sqrt(np.std(x_coords)**2 + np.std(y_coords)**2)
        
        # Recent motion (average velocity over last few frames)
        recent_window = min(8, len(history))
        recent_history = history[-recent_window:]
        
        recent_velocities = []
        for i in range(1, len(recent_history)):
            p1 = np.array(recent_history[i-1])
            p2 = np.array(recent_history[i])
            velocity = np.linalg.norm(p2 - p1) * self.fps
            recent_velocities.append(velocity)
        
        recent_motion = np.mean(recent_velocities) if recent_velocities else 0.0
        
        # Weighted combination favoring recent motion
        total_score = (self.recent_motion_weight * recent_motion + 
                      (1 - self.recent_motion_weight) * historical_motion)
        
        return total_score

    def _calculate_proximity_boost(self, ball_track):
        """Enhanced proximity calculation with better distance handling"""
        if not self.person_tracks or not ball_track['center_history']:
            return 1.0
        
        ball_center = ball_track['center_history'][-1]
        min_distance = float('inf')
        
        for person_track in self.person_tracks:
            if (not person_track['center_history'] or 
                person_track['time_since_update'] > 5): 
                continue
            
            person_center = person_track['center_history'][-1]
            distance = np.linalg.norm(np.array(ball_center) - np.array(person_center))
            min_distance = min(min_distance, distance)
        
        if min_distance <= self.proximity_threshold:
            normalized_distance = min_distance / self.proximity_threshold
            boost = 1.0 + (self.proximity_boost_factor - 1.0) * (1.0 - normalized_distance)**2
            return boost
        
        return 1.0

    def is_action_ball(self, track_id):
        """Check if track ID corresponds to the action ball"""
        return track_id is not None and track_id == self.action_ball_id

    def get_track_velocity(self, track_id):
        """Calculate smoothed velocity with outlier rejection"""
        track = next((t for t in self.ball_tracks + self.person_tracks if t['id'] == track_id), None)
        if not track or len(track['center_history']) < 2:
            return 0.0

        history = track['center_history'][-self.velocity_smoothing_window:]
        if len(history) < 2:
            history = track['center_history']

        velocities = []
        for i in range(1, len(history)):
            p1 = np.array(history[i-1])
            p2 = np.array(history[i])
            displacement = np.linalg.norm(p2 - p1)
            velocity = displacement * self.fps
            velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Remove outliers for more stable velocity
        velocities = np.array(velocities)
        if len(velocities) > 3:
            # Remove values beyond 2 standard deviations
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            mask = np.abs(velocities - mean_vel) <= 2 * std_vel
            velocities = velocities[mask]
        
        return np.mean(velocities) if len(velocities) > 0 else 0.0

    def get_tracking_stats(self):
        """Return current tracking statistics for debugging"""
        return {
            'frame_count': self.frame_count,
            'active_ball_tracks': len([t for t in self.ball_tracks if t['time_since_update'] <= 5]),
            'active_person_tracks': len([t for t in self.person_tracks if t['time_since_update'] <= 5]),
            'action_ball_id': self.action_ball_id,
            'total_ball_tracks_created': self.next_ball_track_id,
            'total_person_tracks_created': self.next_person_track_id - 100
        }

    def force_action_ball(self, track_id):
        """Manually set action ball ID (useful for debugging)"""
        if any(track['id'] == track_id for track in self.ball_tracks):
            self.action_ball_id = track_id
            print(f"Manually set action ball to ID {track_id}")
            return True
        return False