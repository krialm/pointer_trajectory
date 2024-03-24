import cv2
import numpy as np


def empty(_):
    pass



img = cv2.imread('/Users/krialm/Downloads/wetransfer_1_005_250_c001s0001_2024-02-21_1207/1_005_250_C001S0001/1_005_250_c001s0001000001.jpg')



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
    
    th3 = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    kernel = np.ones((7, 7))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    cv2.imshow('Window', imgDil)
    cv2.imshow('Origin', img)
    cv2.imshow('Thresh', th3)

# contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cntArea = cv2.contourArea(cnt)
#     cv2.drawContours(img, cnt, -1, (244, 0, 244), 7)
#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)


    if cv2.waitKey(1) == ord('d'):
        break
cv2.destroyAllWindows()
