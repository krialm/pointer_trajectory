import cv2
import numpy as np

def empty(_):
    pass



def get_contours(frame, thr1=233, thr2=93):


    imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, thr1, thr2)
    
    kernel = np.ones((1,1), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDil, kernel, iterations=2)


    contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours




if __name__ == "__main__":


    img = cv2.imread(r'D:\GitHubRep\pointer_trajectory\croped_test_video.mp4')

    cv2.namedWindow('Parameters') 
    cv2.resizeWindow('Parameters', 640, 150)
    cv2.createTrackbar('Threshold1', 'Parameters', 173, 255, empty)
    cv2.createTrackbar('Threshold2', 'Parameters', 80, 255, empty)

    while True:
        img = cv2.imread(r'/Users/krialm/Projects/pointer_trajectory/out/1_C2_250_C001S0002/new_1_c2_250_c001s0002000001.jpg')
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        
        kernel = np.ones((5, 5), np.uint8)
        imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
        imgErode = cv2.erode(imgDil, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw the contours
        # for contour in contours:
        #     cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        cv2.imshow('Window', img)
        cv2.imshow('Origin', imgErode)

        if cv2.waitKey(1) == ord('d'):
            break

    cv2.destroyAllWindows()
