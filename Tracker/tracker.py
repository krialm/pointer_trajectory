import cv2
import numpy as np
#NOT DONE YET

#The setup:
vid_capture = cv2.VideoCapture('test_vid_1.mp4')
vid_tracker = cv2.Tracker()
colors = np.random.randint(0, 255, (100, 3))
trajectory = []

#The cooking:
ret, frame = vid_capture.read()
boundary_box = cv2.selectROI('Frame', frame, False)
vid_tracker.init(frame, boundary_box)

while True:
    
    if not ret:
        break

    ret, frame = vid_capture.read()
    success, boundary_box = vid_tracker.update(frame)

    if success:
        w1, h1, w2, h2 = [int(i) for i in boundary_box]
        cv2.rectangle(frame, (w1, h1), (w1 + w2, h1 + h2), (0, 255, 0), 2)

        #The pointer:
        pointer = (w1+w2//2, h1+h2//2)
        trajectory.append(pointer)
    
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], colors[i], 2)

    cv2.imshow('Frame', frame)


vid_capture.release()
cv2.destroyAllWindows()