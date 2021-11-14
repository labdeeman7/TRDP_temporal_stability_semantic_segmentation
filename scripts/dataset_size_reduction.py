import cv2
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='create video files from frames')
parser.add_argument("--image_dir_path", type=str, help="path to image folder")
parser.add_argument("--annotation_dir_path", type=str, help="path to annotationfolder")

args = parser.parse_args()
image_dir_path = args.image_dir_path
annotation_dir_path = args.annotation_dir_path

def reduce_image_size(image_dir_path, annotation_dir_path):
   raise NotImplementedError


if __name__ == '__main__':
    reduce_image_size(image_dir_path, annotation_dir_path)  # Calling the generate_video function

