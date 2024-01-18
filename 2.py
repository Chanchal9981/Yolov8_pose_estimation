import os
import cv2
import numpy as np
import random

input_dir = "images9"
output_dir = "images8"

augmentation_params = {
    'rotation_angle': 15,  # Max rotation angle in degrees
    'scale_range': (0.8, 1.2),  # Range for scaling the image
    'flip_probability': 0.5,  # Probability of horizontal flipping
}

for file in os.listdir(input_dir):
    if os.path.splitext(file)[1] == ".txt":
        file1 = open(input_dir + "//" + file, 'r')
        Lines = file1.readlines()
        image = cv2.imread(input_dir + "//" + os.path.splitext(file)[0] + ".jpg")
        image_h, image_w, _ = image.shape
        for line in Lines:
            # print(type(line),line)
            ln_list = line.split(" ")
            ln_list = ln_list[0:-2]
            ln_list = list(map(float, ln_list))
            bbox = ln_list[1:5]
            keypoints = ln_list[5:-1]
            if random.random() < augmentation_params['flip_probability']:
                image = cv2.flip(image, 1)
            rotation_angle = random.uniform(-augmentation_params['rotation_angle'], augmentation_params['rotation_angle'])
            scale_factor = random.uniform(augmentation_params['scale_range'][0], augmentation_params['scale_range'][1])
            rows, cols, _ = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2),rotation_angle, scale_factor)
            image = cv2.warpAffine(image, M, (cols, rows))
            num_keypoints = len(keypoints) // 3
            for i in range(num_keypoints):
                x, y, confidence = keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2]
                new_x = (x - cols / 2) * np.cos(np.radians(rotation_angle)) - (y - rows / 2) * np.sin(np.radians(rotation_angle)) + cols / 2
                new_y = (x - cols / 2) * np.sin(np.radians(rotation_angle)) + (y - rows / 2) * np.cos(np.radians(rotation_angle)) + rows / 2
                new_x = new_x * scale_factor
                new_y = new_y * scale_factor
                keypoints[i * 3] = new_x
                keypoints[i * 3 + 1] = new_y
            x, y, width, height = bbox
            x, y = x * scale_factor, y * scale_factor
            width, height = width * scale_factor, height * scale_factor
            augmented_image_path = os.path.join(output_dir, file.replace(".txt", "_augmented.jpg"))
            cv2.imwrite(augmented_image_path, image)
            with open(os.path.join(output_dir, file), "w") as output_file:
                output_file.write(augmented_image_path + "\n")
                output_file.write(" ".join(map(str, keypoints)) + "\n")
                output_file.write(f"{int(x)} {int(y)} {int(width)} {int(height)}")
print("Augmentation completed.")


