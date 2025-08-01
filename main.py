#!/usr/bin/env python3
"""
Main script to run the processing pipeline
"""

import cv2
import os
from ultralytics import YOLO
import utils

# Constants
INPUT_DIR = "input"
OUTPUT_DIR = "output"
VIDEO_FILE = "test.mp4"
OUTPUT_FILE = "output1.mp4"
MODEL_NAME = "yolov8n.pt"

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
        
        # Draw bounding boxes on the frame
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection
            # Draw rectangle around detected ball
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(annotated_frame, 'Football', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write annotated frame to output video
        out.write(annotated_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")

if __name__ == "__main__":
    main()