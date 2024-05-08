import detection
import cv2
import numpy as np

video_pth = r'D:\GitHubRep\pointer_trajectory\croped_test_video.mp4'

cap = cv2.VideoCapture(video_pth)
_, frame = cap.read()
frame = frame[10:-10, 5:-10]
contours = detection.get_contours(frame, 233, 93)

# the function should return list of points of trajectory
points = your_function()


def calculate_accuracy(points, contours):
    # Initialize counter for points inside the contour
    inside = 0
    for point in points:
        if cv2.pointPolygonTest(contours, (point[0], point[1]), False) >= 0:
            inside += 1
    # Calculate accuracy
    accuracy = inside / len(points)
    return accuracy

while True:

    _, frame = cap.read()
    frame = frame[10:-10, 5:-10]

    
    # Calculate accuracy
    accuracy = calculate_accuracy(points, contours)
    
    # Display the accuracy result on the frame
    cv2.putText(frame, f'Accuracy: {accuracy*100:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('F', frame)
    if cv2.waitKey(0) == ord('d'):
        break

cv2.destroyAllWindows()

