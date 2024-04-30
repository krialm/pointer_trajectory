import detection
import cv2
import numpy as np

video_pth = '/Users/krialm/Projects/pointer_trajectory/croped_test_video.mp4'

cap = cv2.VideoCapture(video_pth)
_, frame = cap.read()
frame = frame[10:-10, 5:-10]
contours = detection.get_contours(frame, 233, 93)




while True:

    _, frame = cap.read()
    frame = frame[10:-10, 5:-10]

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('F', frame)
    if cv2.waitKey(0) == ord('d'):
        break

cv2.destroyAllWindows()

