import cv2
import os

def make_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    if not images:
        print("No images found in the specified folder.")
        return

    images.sort()  # Sort images based on filenames

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))  # Increased frame rate

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Specify the folder containing images and the desired video name
image_folder = '/home/rimantaslav/Desktop/Project_M/test/3_H_1500_C001S0001.cih'
video_name = 'output_video_14.mp4'

# Call the function to create the video
make_video(image_folder, video_name)