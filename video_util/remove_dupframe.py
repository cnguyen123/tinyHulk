#!/usr/bin/env python3
#
#  Copyright (c) 2018-2020 Carnegie Mellon University
#  All rights reserved.
#
# Based on work by Junjue Wang.

#
"""Remove similar frames based on a perceptual hash metric
"""
import os
import random
import shutil

import imagehash
import numpy as np
from PIL import Image
import pandas as pd
import cv2



def checkDiff(image_hash, base_image_hash, threshold):
    if base_image_hash is None:
        return True
    #print("image_hash - base_image_hash =", image_hash - base_image_hash)
    if image_hash - base_image_hash >= threshold:
        return True

    return False


def checkDiffComplete(image_hash, base_image_list, threshold):
    if len(base_image_list) <= 0:
        return True
    for i in base_image_list:
        if not checkDiff(image_hash, i, threshold): #if two images are similar, return false
            return False
    # only return True after checking all images in base_image_list and the checked image are different
    return True


def checkDiffRandom(image_hash, base_image_list, check_ratio, threshold):
    if len(base_image_list) <= 0:
        return True
    check_length = int(len(base_image_list) * check_ratio)
    new_list = []
    new_list.extend(range(len(base_image_list)))
    random.shuffle(new_list)
    for i in new_list[:check_length]:
        if not checkDiff(image_hash, base_image_list[i], threshold):
            return False
    return True


def contProcess(input_folder, threshold):
    base_image_hash = None
    nodup = []
    obj_img_files = os.listdir(input_folder)
    for i in obj_img_files:
        obj_img_file = os.path.join(input_folder, i)

        im = Image.open(obj_img_file)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiff(image_hash, base_image_hash, threshold):
            base_image_hash = image_hash
            nodup.append(i)
    return nodup
def completeProcess(input_folder, threshold):
    base_image_list = []
    nodup2 = []
    obj_img_files = os.listdir(input_folder)
    for i in obj_img_files:
        obj_img_file = os.path.join(input_folder, i)
        im = Image.open(obj_img_file)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiffComplete(image_hash, base_image_list, threshold):
            base_image_list.append(image_hash)
            nodup2.append(i)

    return nodup2

def randomProcess(dic, ratio, threshold):
    if ratio < 0 or ratio > 1:
        raise Exception("Random ratio should between 0 and 1")
    base_image_list2 = []
    nodup3 = []
    print(len(dic["items"]))
    for i in dic["items"]:
        imgpath = i["image"]["path"]
        im = Image.open(imgpath)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiffRandom(image_hash, base_image_list2, ratio, threshold):
            base_image_list2.append(image_hash)
            nodup3.append(i)
    return nodup3


INPUT_FRAME = "../data/frame/"
OUTPUT_REDUCED_FRAME = "../data/reduced_frame/"
CSV_DIR = "../data/csv_data"
obj="Toyplane"
level = 3

DIFF_THRESHOLD = 4
DEFAULT_RATIO = 4
def get_all_steps(input_directory_of_obj):
  '''
  returns all assembly steps of the app (obj)
  '''
  steps = []
  for root, dirnames, filenames in os.walk(input_directory_of_obj):
    if len(dirnames) > 0: steps = dirnames
  return steps
def main():
    input_folder = os.path.join(INPUT_FRAME, obj)
    steps = get_all_steps(input_folder)

    for step in steps:
        nodup_frame = remove_duplicate_frames(os.path.join(input_folder, step), DIFF_THRESHOLD)
        original_csv_file = os.path.join(CSV_DIR, obj) + "/" + step + ".csv"
        new_csv_file = os.path.join(CSV_DIR, obj) + "/" + step +"_ph_{}".format(DIFF_THRESHOLD) + ".csv"
        df = pd.read_csv(original_csv_file)
        new_df = df[df['filename'].isin(nodup_frame)]
        new_df.to_csv(new_csv_file)
        print("-------------")





def duplicates_removal(video_file):
    cap = cv2.VideoCapture(video_file)
    base_image = ""
    nodup2 = []
    count = 0
    success, frame = cap.read()
    output_frame_folder = os.path.join("../data/frame/out/")
    threshold = 1
    base_img_hash = [""]
    while success:
        saved_frame_files = os.listdir(output_frame_folder)
        name = "frame%05d.jpg" % count
        print(name)
        if len(saved_frame_files) == 0:
            file_name = os.path.join(output_frame_folder, name)
            cv2.imwrite(file_name, frame)  # save frame as JPG file
            count += 1
        else:
            frame_files = os.listdir(output_frame_folder)
            frame_files.sort()

            base_im = Image.open(os.path.join(output_frame_folder,frame_files[len(frame_files) -1]))
            base_im_arr = np.asarray(base_im)
            im = Image.fromarray(base_im_arr)
            base_img_hash = imagehash.phash(im)


            current_im_arr = np.asarray(frame)
            c_im = Image.fromarray(current_im_arr)
            current_img_hash = imagehash.phash(c_im)
            if checkDiffComplete(current_img_hash, [base_img_hash], threshold):
                file_name = os.path.join(output_frame_folder, name)
                cv2.imwrite(file_name, frame)  # save frame as JPG file
                count += 1
            else:
                print("duplicate, not save!")

        success, frame = cap.read()






def remove_duplicate_frames(input_folder, threshold):
    print("Removing duplicated frames in ", input_folder, "...")
    nodup_frames = completeProcess(input_folder, threshold)
    all_frame_files = os.listdir(input_folder)

    removed_frame_files = list(set(all_frame_files) - set(nodup_frames))
    print("Completed!")
    print("Total frames removed:", len(removed_frame_files))
    print("Total frames remained (no duplicated):",len(nodup_frames))
    #for i in removed_frame_files:
    #    os.remove(os.path.join(input_folder, i))
    return nodup_frames
if __name__ == "__main__":

    main()
    #duplicates_removal("../data/video/0wheel/VID_20221008_113643.mp4")