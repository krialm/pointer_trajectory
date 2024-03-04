import cv2
import numpy as np


def empty(_):
    pass



img = cv2.imread('new/1_005_250_C001S0001/new_1_005_250_c001s0001000001.jpg')



cv2.namedWindow('Parameters') 
cv2.resizeWindow('Parameters', 640, 150)
cv2.createTrackbar('Threshold1', 'Parameters', 56, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 91, 255, empty)




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

    # Draw the largest contour on the original image
    result = cv2.drawContours(img.copy(), [largest_contour], -1, (0, 0, 255), 2)

    th3 = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    kernel = np.ones((7, 7))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    cv2.imshow('Window', imgDil)
    cv2.imshow('Origin', img)
    cv2.imshow('Thresh', th3)
    cv2.imshow('Result', result)


    if cv2.waitKey(1) == ord('d'):
        break
cv2.destroyAllWindows()