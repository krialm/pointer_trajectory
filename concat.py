import detection
import numpy as np
import cv2
import math

def find_nearest_point(point, contours):
  """
  This function finds the nearest point on a trajectory (contour) to a given point.

  Args:
      point: A tuple or list representing the point (x, y coordinates).
      contours: A list of NumPy arrays, where each array represents a contour (sequence of points).

  Returns:
      A tuple containing:
          - The nearest point on the trajectory (x, y coordinates).
          - The index of the contour containing the nearest point.
          - The minimum distance between the point and the nearest point on the trajectory.

      - None if no contours are provided.
  """
  if not contours:
      return None  # No contours

  min_distance = float('inf')  # Initialize minimum distance to infinity
  nearest_point = None

  for i in contours:
      for contour in i:
        # Calculate distances between point and each point in the contour
        distances = np.linalg.norm(contour - point, axis=1)  # Euclidean distance
        min_idx = np.argmin(distances)  # Index of the closest point on the current contour

        # Update minimum distance and nearest point if closer point found
        if np.any(distances < min_distance):
            min_distance = distances[min_idx]
            nearest_point = contour[min_idx]
            nearest_contour_index = i

  return nearest_point

def check_distance(point, trajectory):
    nearest_point = find_nearest_point(point, trajectory)
    # Calculate squared difference for each dimension
    squared_differences = [(p1 - p2) ** 2 for p1, p2 in zip(point, nearest_point)]

    # Sum the squared differences
    sum_of_squares = sum(squared_differences)

    # Take the square root to get the Euclidean distance
    distance = math.sqrt(sum_of_squares)

    return distance



def detect_trajectory(file_path):
    """" 
    This function takes first frame 
    and use function get_contours
    from detection.py to create 
    gloabal variable of trajectory contours
    """
    crap = cv2.VideoCapture(file_path)
    done, frame = crap.read()
    global trajectory_contours
    trajectory_contours = detection.get_contours(frame, thr1=100, thr2=140)


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


    done, cur_frame = cap.read()

    done, next_frame = cap.read()

    centers = []

    roi = (0, 0, frame_w, frame_h)

    half_length_frame = total_frames // 2

    all_frames = 0
    good_frames = 0

    while cap.isOpened():
        if done:
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            #Threshold adjustment
            if current_frame_num >= half_length_frame:
                threshold = 15
            else:
                threshold = 15

            diff = cv2.absdiff(cur_frame, next_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(binary, None, iterations=10)
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #Update ROI based on the latest added point
            #if len(centers) > 30:
            #    latest_point = centers[-1]
            #    roi = (max(0, latest_point[0] - 10), max(0, latest_point[1] - 10),
            #        min(frame_w, latest_point[0] + 10), min(frame_h, latest_point[1] + 10))
            #    if current_frame_num >= half_length_frame:
            #        roi = (max(0, latest_point[0]-12), max(0, latest_point[1]-12),
            #            min(frame_w, latest_point[0]+12), min(frame_h, latest_point[1]+12))

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
                        
                        cv2.drawContours(cur_frame, trajectory_contours, -1, (0, 255, 0), 1)
                        cv2.line(cur_frame, current_point, closest_point, (0, 0, 255), 2)


            
            distance = check_distance(current_point, trajectory_contours)
            all_frames+=1
            if distance >= 0 and distance <7:
                good_frames+=1
            cv2.putText(cur_frame, f"Accuracy: {str(round(good_frames/all_frames, 3))}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

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
    with open("coordinates.txt", "w") as f:
        for i in centers:
            f.write(str(i)[1:-1] + "\n")
        f.close()

if __name__ == '__main__':

    draw_trajectory(r"D:\GitHubRep\pointer_trajectory\Tracking\output_video_8-ezgif.com-crop-video (1).mp4")
