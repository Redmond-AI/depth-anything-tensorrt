import cv2
import numpy as np
import sys

def calculate_flicker(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Initialize variables
    prev_frame = None
    total_diff = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate absolute difference between current and previous frame
            diff = cv2.absdiff(gray, prev_frame)
            total_diff += np.mean(diff)
            frame_count += 1

        prev_frame = gray

    cap.release()

    if frame_count == 0:
        return None

    # Calculate average difference per frame
    avg_diff = total_diff / frame_count
    return avg_diff

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python flicker_test.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    flicker_score = calculate_flicker(video_path)

    if flicker_score is not None:
        print(f"Flicker score: {flicker_score:.2f}")
    else:
        print("Error: Could not calculate flicker score")
