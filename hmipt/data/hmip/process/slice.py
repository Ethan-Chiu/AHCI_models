import cv2
import os


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


def find_avi_files(directory):
    avi_files = []
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.avi'):
                # Create the relative path
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                avi_files.append(relative_path)
    return avi_files


process_dir = "../../video/"

videos = find_avi_files(process_dir)
print(videos)

output_dir = "../../frames/"

for v in videos:
    try:
        input_file = os.path.join(process_dir, v)
        output_folder = os.path.dirname(os.path.join(output_dir, v))
        print(output_folder)
        slice_avi_to_frames(input_file, output_folder)
    except Exception as e:
        print(e)
        pass