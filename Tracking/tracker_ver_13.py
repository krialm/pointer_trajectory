def draw_trajectory(file_path):
    import cv2
    import numpy as np

    vid = file_path
    cap = cv2.VideoCapture(vid)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "Detected_Motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    done, cur_frame = cap.read()
    done, next_frame = cap.read()

    centers = []

    # Initializing ROI as the entire frame
    roi = (0, 0, frame_w, frame_h)

    # Find the frame at half the video length
    half_length_frame = total_frames // 2

    while cap.isOpened():
        if done:
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            #Threshold adjustment
            if current_frame_num >= half_length_frame:
                threshold = 11
            else:
                threshold = 15

            diff = cv2.absdiff(cur_frame, next_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(binary, None, iterations=10)
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Update ROI based on the latest added point
            if len(centers) > 30:
                latest_point = centers[-1]
                roi = (max(0, latest_point[0] - 50), max(0, latest_point[1] - 50),
                    min(frame_w, latest_point[0] + 50), min(frame_h, latest_point[1] + 50))
                if current_frame_num >= half_length_frame:
                    roi = (max(0, latest_point[0]-12), max(0, latest_point[1]-12),
                        min(frame_w, latest_point[0]+12), min(frame_h, latest_point[1]+12))

            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                if cv2.contourArea(cnt) < 50 or not (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]):
                    continue

                cv2.rectangle(cur_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

                center_x = x + w // 2
                center_y = y + h // 2

                centers.append((center_x, center_y))

            for i in range(len(centers)):
                current_point = centers[i]
                min_distance = float('inf')
                closest_point = None
                for j in range(max(0, i - 5), i):
                    distance = np.sqrt((current_point[0] - centers[j][0])**2 + (current_point[1] - centers[j][1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = centers[j]
                if closest_point is not None:
                    line_length = np.sqrt((current_point[0] - closest_point[0])**2 + (current_point[1] - closest_point[1])**2)
                    max_line_length = 55
                    if line_length < max_line_length:
                        cv2.line(cur_frame, current_point, closest_point, (0, 0, 255), 2)

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

draw_trajectory(r"D:\GitHubRep\pointer_trajectory\Tracking\test_vid_1.mp4")