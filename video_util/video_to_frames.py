#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Chanh Nguyen
# Created Date: 02/02/22
# version ='1.0'
# ---------------------------------------------------------------------------
"""Parsing a video to frames. \n
Note:
  - Change VIDEO_INPUT_DIR and FRAME_OUT_DIR to the according directories where videos locate.
  - Change obj to the folder's name, for example longbar
"""
# ---------------------------------------------------------------------------

import numpy as np
from PIL import Image
import cv2
import os
from configparser import ConfigParser
import imagehash
VIDEO_INPUT_DIR = ""
FRAME_OUTPUT_DIR = ""
obj = ""
import time

DIFF_THRESHOLD = 1
def is_dissimilar(image_hash, base_image_hash, threshold):
  if base_image_hash is None:
    return True
  # print("image_hash - base_image_hash =", image_hash - base_image_hash)
  if image_hash - base_image_hash >= threshold:
    return True

  return False


def get_all_steps(input_directory_of_obj):
  '''
  returns all assembly steps of the app (obj)
  '''
  steps = []
  for root, dirnames, filenames in os.walk(input_directory_of_obj):
    if len(dirnames) > 0: steps = dirnames
  print(steps)
  return steps

def video_to_frames():

  steps = get_all_steps(os.path.join(VIDEO_INPUT_DIR, obj))
  #steps = ['wing_bolt']
  for step in steps:
    start_time = time.time()
    print('parsing videos to frames for step ', step)
    output_folder = os.path.join(FRAME_OUTPUT_DIR, obj, step)
    # Check if the specified path exists
    isExist = os.path.exists(output_folder)
    if not isExist:
      os.makedirs(output_folder)


    unique_count = 0
    unique_frames = []
    last_unique_frame_phash = None
    videos_list = os.listdir(os.path.join(VIDEO_INPUT_DIR, obj, step))
    count = 0
    for v in videos_list:
      print("video", v, "...")
      video_file = os.path.join(VIDEO_INPUT_DIR, obj,step, v)
      vidcap = cv2.VideoCapture(video_file)
      success, image = vidcap.read()

      while success:

        cur_image_phash = imagehash.phash(Image.fromarray(np.asarray(image)))
        #if is_dissimilar(cur_image_phash, last_unique_frame_phash, DIFF_THRESHOLD):
        if True:
          unique_count += 1
          unique_frames.append(count)
          last_unique_frame_phash = cur_image_phash
          name = "frame%05d.jpg" % count
          file_name = os.path.join(output_folder, name)
          cv2.imwrite(file_name, image)  # save frame as JPG file


        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

    print("Total: {} Unique: {} Removed: {}".format(count, unique_count, count - unique_count))
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Completed! All frames are saved at ", output_folder)




def main():


  # Read config.ini file
  config_object = ConfigParser()
  config_object.read("../config.ini")

  # Get all necessity paths
  labeling_directory = config_object["LABELING_DIRECTORY"]
  global VIDEO_INPUT_DIR
  VIDEO_INPUT_DIR = labeling_directory["VIDEO_INPUT_DIR"]
  global FRAME_OUTPUT_DIR
  FRAME_OUTPUT_DIR = labeling_directory["FRAME_DIR"]

  dataset_name = config_object["DATASET_NAME"]
  global obj
  obj= dataset_name["obj"]
  video_to_frames()


if __name__ == '__main__':
   main()



