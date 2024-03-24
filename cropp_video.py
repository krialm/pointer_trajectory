import cv2 



cap = cv2.VideoCapture('/Users/krialm/Projects/pointer_trajectory/new_1_005_250_c001s0001000001.mp4')

_, frame = cap.read()
r = cv2.selectROI('Frame', frame, False)
frame_width = int(r[1])+int(r[1]+r[3])
frame_height = int(r[0])+int(r[0]+r[2])
# Create VideoWriter object
result = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    
while True:
    _, frame = cap.read()

    sky = frame[int(r[1]):int(r[1]+r[3]),  
                      int(r[0]):int(r[0]+r[2])] 

    cv2.imshow('d', sky)
    result.write(sky)

    if cv2.waitKey(1) == ord("f"):
        break

result.release()
cv2.destroyAllWindows()
cap.release()