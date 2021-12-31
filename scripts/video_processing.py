import cv2
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='create video files from frames')
parser.add_argument("--image_folder_path", type=str, help="path to folder")
parser.add_argument("--video_path", type=str, help="path to video")

args = parser.parse_args()
image_folder_path = args.image_folder_path
video_path = args.video_path


def generate_video(image_folder, video_path):
    """
    Generate videos from folder of images.
    :param image_folder: folder path
    :param video_path: name for video
    :return: none
    """
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png") or
              img.endswith("tif")]

    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

    # Appending the images to the video one by one
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image), cv2.IMREAD_UNCHANGED)
        video.write(img)

    # Deallocate memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


if __name__ == '__main__':
    generate_video(image_folder_path, video_path)  # Calling the generate_video function

