from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import os

#The setup:
vid_capture = cv2.VideoCapture('D:\GitHubRep\pointer_trajectory\croped_test_video.mp4')
vid_tracker = cv2.TrackerKCF_create()
colors = np.random.randint(0, 255, (100, 3))
trajectory = []

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="D:\GitHubRep\pointer_trajectory\croped_test_video.mp4", help="path to the video file")
args = vars(ap.parse_args())
video_path = args["video"]

# Does vid exist?
if not os.path.exists(video_path):
    print(f"Error: The video file '{video_path}' does not exist.")
    exit()

# Initialize the video
vs = cv2.VideoCapture(video_path)

# Check if vid is opened
if not vs.isOpened():
    print(f"Error: Failed to open video file '{video_path}'.")
    exit()


# Lower and upper boundaries for the gray color
lower_gray = 50
upper_gray = 200

# Initialize the deque to store the points
pts = deque(maxlen=64)

# Loop over frames from the vid
while True:
    # Read the next frame from the vid
    ret, frame = vs.read()

    if not ret:
        break

    # Resize the frame and convert it to gray
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Construct a mask for the gray color
    mask = cv2.inRange(gray, lower_gray, upper_gray)

    # Find contours in the mask and initialize the current (x, y) center of the object
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # If contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        print(f"Center: {center}, Radius: {radius}")

        if radius > 10:
            # Draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Update the points 
    pts.appendleft(center)

    # Loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        # Draw:
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            

    # Show frame 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q = stop loop
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
