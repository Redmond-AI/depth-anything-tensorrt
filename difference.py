import cv2
import numpy as np
import argparse

def create_difference_video(video1_path, video2_path, output_path):
    # Open input videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Calculate absolute difference between frames
        diff = cv2.absdiff(frame1, frame2)
        diff*=5

        # Write difference frame to output video
        out.write(diff)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a difference video from two input videos.")
    parser.add_argument("--input1", required=True, help="Path to the first input video")
    parser.add_argument("--input2", required=True, help="Path to the second input video")
    parser.add_argument("--output", required=True, help="Path for the output difference video")
    
    args = parser.parse_args()

    create_difference_video(args.input1, args.input2, args.output)
    print(f"Difference video created: {args.output}")
