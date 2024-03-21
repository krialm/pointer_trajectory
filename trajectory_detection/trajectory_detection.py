import cv2
import numpy as np


def empty(_):
    pass


def is_belong_to_trajectory(pointer_coordinate):

    if pointer_coordinate in largest_contour:
        return True
    
    return False

img = cv2.imread('/Users/krialm/Projects/pointer_trajectory/samples/cropped_traj.jpg')

cv2.namedWindow('Parameters') 
cv2.resizeWindow('Parameters', 640, 150)
cv2.createTrackbar('Threshold1', 'Parameters', 60, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 74, 255, empty)

pix_len = 48.3/510

print(pix_len)

while True:

    threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area, shape, etc.
    # For simplicity, let's assume the largest contour is the curve
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the area of the largest contour
    area = cv2.contourArea(largest_contour)

    # Draw the largest contour and display the area on the original image
    result = cv2.drawContours(img.copy(), [largest_contour], -1, (0, 0, 255), 2)
    cv2.putText(result, f'Area: {area}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Result', result)

    if cv2.waitKey(1) == ord('d'):
        break

cv2.destroyAllWindows()
