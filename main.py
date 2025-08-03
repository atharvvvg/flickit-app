#!/usr/bin/env python3
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
MODEL_NAME = "yolov8n.pt"

# Visualization constants
TRAIL_LENGTH = 20  # Number of points to show in action ball trail
TRAIL_COLORS = [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100)]  # Fading trail colors
ACTION_BALL_COLOR = (0, 255, 255)  # Cyan for action ball
STATIONARY_BALL_COLOR = (0, 255, 0)  # Green for stationary balls
TEXT_COLOR = (255, 255, 0)  # Light blue/cyan for text
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
BOX_THICKNESS = 2

def draw_action_ball(frame, track, tracker):
    """Draw action ball with trail and center point"""
    x1, y1, x2, y2 = track['bbox']
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Draw trail (last TRAIL_LENGTH positions)
    if 'center_history' in track and len(track['center_history']) > 1:
        trail_points = track['center_history'][-TRAIL_LENGTH:]  # Last TRAIL_LENGTH points
        
        # Draw trail lines with fading effect
        for i in range(len(trail_points) - 1):
            # Calculate fading alpha based on position in trail
            alpha = (i + 1) / len(trail_points)
            color_idx = min(int(alpha * len(TRAIL_COLORS)), len(TRAIL_COLORS) - 1)
            color = TRAIL_COLORS[color_idx]
            thickness = max(1, int(3 * alpha))
            
            pt1 = (int(trail_points[i][0]), int(trail_points[i][1]))
            pt2 = (int(trail_points[i + 1][0]), int(trail_points[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw center circle
    cv2.circle(frame, (center_x, center_y), 8, ACTION_BALL_COLOR, -1)
    cv2.circle(frame, (center_x, center_y), 8, (0, 0, 0), 2)  # Black border
    
    # Calculate velocity
    velocity = tracker.get_track_velocity(track['id'])
    
    # Draw annotation text
    text = f"ID {track['id']} | ACTION | V={velocity:.1f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y2 + 25
    
    # Draw text background
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Draw text
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
    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
    
    # Calculate velocity
    velocity = tracker.get_track_velocity(track['id'])
    
    # Draw annotation text
    text = f"ID {track['id']} | STATIONARY | V={velocity:.1f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y2 + 25
    
    # Draw text background
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

def main():
    """Main function to process video"""
    # Construct input and output paths
    input_path = os.path.join(INPUT_DIR, VIDEO_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Check if input video exists
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found!")
        print(f"Please place {VIDEO_FILE} in the {INPUT_DIR}/ directory.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Open video file
    video = cv2.VideoCapture(input_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file '{output_path}'")
        video.release()
        return
    
    print(f"Saving output to: {output_path}")
    
    # Initialize tracker
    tracker = Tracker()
    tracker.set_fps(fps)
    
    frame_count = 0
    
    # Process all frames
    while True:
        # Read frame
        ret, frame = video.read()
        
        # Break if no more frames
        if not ret:
            print("Finished processing all frames")
            break
        
        frame_count += 1
        
        # Detect balls in current frame
        detections = utils.detect_balls(frame, model)
        
        # Update tracker with detections
        tracks = tracker.update(detections)
        
        # Draw dynamic visualization based on ball classification
        annotated_frame = frame.copy()
        
        for track in tracks:
            # Check if this is the action ball
            if tracker.is_action_ball(track['id']):
                draw_action_ball(annotated_frame, track, tracker)
            else:
                draw_stationary_ball(annotated_frame, track, tracker)
        
        # Write annotated frame to output video
        out.write(annotated_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
            print(f"Active tracks: {len(tracks)}")
    
    # Cleanup
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")

if __name__ == "__main__":
    main()