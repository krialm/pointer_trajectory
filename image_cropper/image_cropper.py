import os
import cv2


image_dir = 'Data'
cropped_image_dir = 'Cropped Data'


for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    if os.path.isdir(filepath):
        new_dir = os.path.join(cropped_image_dir, filename)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for image_filename in os.listdir(filepath):
            if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
                img = cv2.imread(os.path.join(filepath, image_filename))
                cropped_image = img[0:1025, 95:605]
                cv2.imwrite(os.path.join(new_dir, 'new_' + image_filename), cropped_image)