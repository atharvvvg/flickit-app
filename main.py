"""
Main script to run the processing pipeline
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
import utils
from tracker import Tracker

# Constants
INPUT_DIR = "input"
OUTPUT_DIR = "output"
VIDEO_FILE = "test.mp4"
OUTPUT_FILE = "output1.mp4"
MODEL_NAME = "yolov8s.pt"

# Visualization constants
TRAIL_LENGTH = 20
TRAIL_COLORS = [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100)]
ACTION_BALL_COLOR = (0, 255, 255)
STATIONARY_BALL_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 0)
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
BOX_THICKNESS = 2

def draw_action_ball(frame, track, tracker):
    """Draw action ball with trail and center point"""
    x1, y1, x2, y2 = track['bbox']
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Draw trail 
    if 'center_history' in track and len(track['center_history']) > 1:
        trail_points = track['center_history'][-TRAIL_LENGTH:]
        
        for i in range(len(trail_points) - 1):
            alpha = (i + 1) / len(trail_points)
            color_idx = min(int(alpha * len(TRAIL_COLORS)), len(TRAIL_COLORS) - 1)
            color = TRAIL_COLORS[color_idx]
            thickness = max(1, int(3 * alpha))
            
            pt1 = (int(trail_points[i][0]), int(trail_points[i][1]))
            pt2 = (int(trail_points[i + 1][0]), int(trail_points[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw center circle
    cv2.circle(frame, (center_x, center_y), 8, ACTION_BALL_COLOR, -1)
    cv2.circle(frame, (center_x, center_y), 8, (0, 0, 0), 2)
    
    velocity = tracker.get_track_velocity(track['id'])
    
    # Draw annotation text
    text = f"ID {track['id']} | ACTION"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y2 + 25
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

def draw_stationary_ball(frame, track, tracker):
    """Draw stationary ball with bounding box and ID"""
    x1, y1, x2, y2 = track['bbox']
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), STATIONARY_BALL_COLOR, BOX_THICKNESS)
    
    # Draw center dot
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    velocity = tracker.get_track_velocity(track['id'])
    
    # Draw annotation text
    text = f"ID {track['id']} | STATIONARY | V={velocity:.1f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y2 + 25
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

def draw_person(frame, track, tracker):
    """Draw person with bounding box and ID"""
    x1, y1, x2, y2 = track['bbox']
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), BOX_THICKNESS)
    
    # Draw center dot
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 165, 255), -1)
    
    velocity = tracker.get_track_velocity(track['id'])
    
    # Draw annotation text
    text = f"Person ID {track['id']} | V={velocity:.1f}"
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y2 + 25
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 165, 255), TEXT_THICKNESS)

def main():
    input_path = os.path.join(INPUT_DIR, VIDEO_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found!")
        print(f"Please place {VIDEO_FILE} in the {INPUT_DIR}/ directory.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading YOLO model: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file '{output_path}'")
        video.release()
        return
    
    print(f"Saving output to: {output_path}")
    
    tracker = Tracker()
    tracker.set_fps(fps)
    
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("Finished processing all frames")
            break
        
        frame_count += 1
        
        ball_detections, person_detections = utils.detect_objects(frame, model)
        ball_tracks, person_tracks = tracker.update(ball_detections, person_detections)
        annotated_frame = frame.copy()
        
        for track in ball_tracks:
            if tracker.is_action_ball(track['id']):
                draw_action_ball(annotated_frame, track, tracker)
            else:
                draw_stationary_ball(annotated_frame, track, tracker)
        
        for track in person_tracks:
            draw_person(annotated_frame, track, tracker)
        
        out.write(annotated_frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
            print(f"Active ball tracks: {len(ball_tracks)}, Active person tracks: {len(person_tracks)}")

    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")

if __name__ == "__main__":
    main()