# Detect-QR_Autonomous-Drone
This project processes a video to detect ArUco markers and calculate their poses. The processed video, along with detected markers data, is saved to an output file.

## Features
1) Detects ArUco markers in video frames.
2) Estimates the pose of detected markers (translation and rotation).
3) Saves the processed video with annotated markers.
4) Outputs a CSV file containing detailed information about detected markers, including their 2D coordinates, distance, and Euler angles (yaw, pitch, roll).

## Usage
1) Clone the Repository.
2) Install the Required Dependencies like : numpy, pandas, Cv2.
3) Place your input video in the project directory. The default input video name is GoPro.mp4.
4) Run the script.
5) Check the output: The processed video and CSV file will be saved in the current directory with the specified output names.

## Project Structure
1) qr_camera.py: Main script for processing the video and detecting ArUco markers.
2) dependencies required like : numpy, pandas, Cv2.
3) GoPro.mp4: (Example) Input video file to be placed in the project directory.

## General Explanation
The qr_camera.py script uses OpenCV (cv2) to process a video (GoPro.mp4 by default) and detect ArUco markers in each frame. It calculates the pose (translation and rotation) of detected markers, annotates them on the video frames, and saves the processed video with annotations. Additionally, it outputs a CSV file containing detailed information about each detected marker's position and orientation throughout the video. The script is structured to handle video input/output, marker detection, pose estimation, and data collection efficiently.

## Output
### CSV Output
The CSV file contains detailed information about the detected markers. Each row in the CSV file corresponds to a detected marker and includes the following columns:
- Frame_ID: The ID of the video frame in which the marker was detected.
- QR_ID: The ID of the detected QR code.
- QR_2D_Coordinates: The 2D coordinates of the QR code corners.
- Distance: The distance of the QR code from the camera.
- Yaw: The yaw angle (rotation around the vertical axis) of the QR code.
- Pitch: The pitch angle (rotation around the lateral axis) of the QR code.
- Roll: The roll angle (rotation around the longitudinal axis) of the QR code.

### Output video
The output video is a processed version of the input video where each frame is annotated with detected ArUco markers.
The annotations include marker boundaries and IDs. This annotated video is saved to the specified output path (e.g., output_video.mp4).

## Authors
- Evyatar Yosef - 207467820
- Hai Levi 313589038

 













