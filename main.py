import detection
import numpy as np
import cv2
import math
import datetime
import openpyxl
import os

def save_data_to_excel(file_path, distance, pointer_coordinate, closest_point):
    """
    Save results to an Excel file. Create the file if it doesn't exist.
    
    Parameters:
    file_path (str): The path to the Excel file.
    results (list of tuples): A list of tuples where each tuple contains:
        - distance (float): The distance between the pointer and the closest point.
        - pointer_coordinate (tuple): The (x, y) coordinates of the pointer.
        - closest_point (tuple): The (x, y) coordinates of the closest point.
    """
    file_exists = os.path.isfile(file_path)
    
    if file_exists:
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Timestamp", "Distance (in mm)", "Data Point X", "Data Point Y", "Closest Point X", "Closest Point Y"])
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_row = [timestamp, distance, pointer_coordinate[0], pointer_coordinate[1], closest_point[0], closest_point[1]]
    sheet.append(data_row)
    workbook.save(file_path)

def find_nearest_point(point, contours):
    """
    Finds the nearest point on a trajectory (contour) to a given point.
    """
    if not contours:
        return None
    
    min_distance = float('inf')
    nearest_point = None

    for contour in contours:
        for pt in contour:
            pt = pt[0]
            distance = np.linalg.norm(pt - point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = pt

    return nearest_point

def check_distance(point, trajectory):
    """
    Calculate the Euclidean distance between a point and its nearest point on the trajectory.
    """
    nearest_point = find_nearest_point(point, trajectory)
    if nearest_point is None:
        return float('inf')

    distance = np.linalg.norm(np.array(point) - np.array(nearest_point))
    return distance * 0.095

def detect_trajectory(file_path):
    """
    Create a global variable of trajectory contours from the first frame of the video.
    """
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the video")
    global trajectory_contours
    trajectory_contours = detection.get_contours(frame, thr1=100, thr2=140)
    cap.release()

def draw_trajectory(file_path):
    vid = file_path
    cap = cv2.VideoCapture(vid)

    detect_trajectory(vid)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "Detected_Motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    ret, cur_frame = cap.read()
    ret, next_frame = cap.read()

    centers = []
    roi = (0, 0, frame_w, frame_h)
    half_length_frame = total_frames // 2
    all_frames = 0
    good_frames = 0

    while cap.isOpened():
        if ret:
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            threshold = 15

            diff = cv2.absdiff(cur_frame, next_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(binary, None, iterations=10)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
                    distance = np.linalg.norm(np.array(current_point) - np.array(centers[j]))
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = centers[j]
                if closest_point is not None:
                    line_length = np.linalg.norm(np.array(current_point) - np.array(closest_point))
                    max_line_length = 55
                    if line_length < max_line_length:
                        cv2.drawContours(cur_frame, trajectory_contours, -1, (0, 255, 0), 1)
                        cv2.line(cur_frame, current_point, closest_point, (0, 0, 255), 2)

            distance = check_distance(current_point, trajectory_contours)
            save_data_to_excel('Results.xlsx', round(distance, 3), current_point, find_nearest_point(current_point, trajectory_contours))

            all_frames += 1
            if distance >= 0 and distance < 0.7:
                good_frames += 1

            cv2.putText(cur_frame, f"Accuracy: {round(good_frames/all_frames, 3)}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(cur_frame, f"Distance: {round(distance, 3)} mm", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("frame", cur_frame)
            output.write(cur_frame)

            cur_frame = next_frame
            ret, next_frame = cap.read()

            if cv2.waitKey(30) == ord("f"):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    output.release()

    with open("coordinates.txt", "w") as f:
        for i in centers:
            f.write(str(i)[1:-1] + "\n")

if __name__ == '__main__':
    draw_trajectory(r"/Users/krialm/Projects/pointer_trajectory/Tracking/output_video_3.mp4")
