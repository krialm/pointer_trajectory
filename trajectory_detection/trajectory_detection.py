import cv2

vid = r""#insert vid path

cap = cv2.VideoCapture(vid)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"X264")
path = r".../Detected Motion.MP4"#insert detected path
output = cv2.VideoWriter(path, fourcc, 30, (frame_w, frame_h))

done, cur_frame = cap.read()
done, next_frame = cap.read()

img = cv2.imread('/Users/krialm/Projects/pointer_trajectory/samples/cropped_traj.jpg')

while cap.isOpened():
    if done:
        diff = cv2.absdiff(cur_frame, next_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh, binary = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(binary, None, iterations=12)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            
            if cv2.contourArea(cnt) < 1000:
                continue
            
            cv2.rectangle(cur_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        for i in range(1, len(centers)):
            cv2.line(cur_frame, centers[i-1], centers[i], (0, 0, 255), 2)
        
        cv2.imshow("frame", cur_frame)
        output.write(cur_frame)
        
        cur_frame = next_frame
        done, next_frame = cap.read()
        
        if cv2.waitKey(30) == ord("f"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
output.release()

