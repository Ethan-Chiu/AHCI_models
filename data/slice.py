import cv2
import os
import sys

# label = sys.argv[1]
# index = sys.argv[2]
labels = "bottle cup mouse keyboard phone".split()
names = ["test", *range(1, 11)]

def slice_avi_to_frames(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(input_file)
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Total {frame_count} frames extracted.")

for label in labels:
    for name in names:
        try:
            input_file = f"video/{label}/{label}_{name}.avi"
            output_folder = f"frames/{label}/{name}"
            slice_avi_to_frames(input_file, output_folder)
        except Exception as e:
            # print(e)
            pass