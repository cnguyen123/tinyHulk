#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Chanh Nguyen
# Created Date: 02/02/22
# version ='1.0'
# ---------------------------------------------------------------------------
"""Drawing a rectangle bounding box around an object in the input frame. \n
Note:
  - OUTPUT_DIR: where the annotated frames are saved
  - INPUT_DIR: where the frames locate
  - CSV_DIR: where the csv file (the file keeps information of frames and its boundingbox) locates
"""
# ---------------------------------------------------------------------------
import shutil
import cv2
import numpy as np
import os
from pathlib import Path
import csv

from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format
from configparser import ConfigParser
import time
OUTPUT_DIR = ""
INPUT_DIR = ""
CSV_DIR = ""
LABEL_MAP_DIR = ""
ABNORMAL_ANNOTATED_FRAME_DIR = ""
#obj_ = ['step0', 'step1','step2', 'step3', 'step4', 'step5', 'step6', 'step7']
obj_ =""

def detect_objects(frame):
    """Detect the possible objects in an input frame \n
    Input: frame \n
    Output: countours around the possible objects in the input frame"""

    # Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("mask", mask)
    objects_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
            objects_contours.append(cnt)

    return objects_contours


def drawboundingbox(frame_filename, padding=30):
    """ Drawing bounding box (i.e., rectangle) around the input frame. \n
    Input:
    - frame_filename: the full filename of the input frame \n
    - padding: an amount of padding to the bounding box
    Output: annotated frame, the frame's shape, coordinate of the bounding box (xmin, ymin, xmax, ymax) """

    img = cv2.imread(frame_filename)
    contours = detect_objects(img)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    dimension_img = img.shape
    x_min = x - padding
    y_min = y- padding
    x_max = x + w + padding
    y_max = y + h + padding
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    return img, dimension_img[0], dimension_img[1], x_min, y_min, x_max, y_max
    #print(x,y, w,h )
    #img = cv2.resize(img, (700,700))
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)

def is_contour_in_edge(frame, cnt):
    """
    Check if the current contour covers the edge of the frame
    :param frame: the current frame
    :param cnt: the current countour
    :return: True if the contour keeps the edge
    """
    fr_shape = frame.shape
    ymax = fr_shape[0]
    xmax = fr_shape[1]
    ymin = 0
    xmin = 0
    delta = 5
    x, y, w, h = cv2.boundingRect(cnt)
    if (x - delta) <= xmin or (y - delta) <= ymin or (y + h + delta) >= ymax or (x + w + delta)>= xmax:
        return True

    return False




def drawboundingbox2(frame_filename, padding=30, maxArea = 500):
    frame = cv2.imread(frame_filename)
    dimension_img = frame.shape
    #Convert Image to grayscale
    #frame = cv2.GaussianBlur(frame, (7, 7), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    #cv2.imshow("mask", mask)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("mask", mask)
    objects_contours = []

    X_MIN = 10000
    Y_MIN = 10000
    X_MAX = -1000
    Y_MAX = -1000
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > maxArea:
            if is_contour_in_edge(frame, cnt):
                continue

            x, y, w, h = cv2.boundingRect(cnt)


            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            if X_MIN > x_min:
                X_MIN = x_min
            if Y_MIN > y_min:
                Y_MIN = y_min
            if X_MAX < x_max:
                X_MAX = x_max
            if Y_MAX < y_max:
                Y_MAX = y_max

    #frame = cv2.circle(frame, (X_MIN, Y_MIN), radius=10, color=(255, 0, 255), thickness=-1)
    #frame = cv2.circle(frame, (X_MAX, Y_MAX), radius=10, color=(255, 0, 255), thickness=-1)

    xmi = min(max(X_MIN -padding,0), dimension_img[1])
    ymi = min(max(Y_MIN- padding,0), dimension_img[0])
    xma = max(min(X_MAX + padding,dimension_img[1]), 0)
    yma = max(min(Y_MAX + padding, dimension_img[0]), 0)
    cv2.rectangle(frame, (xmi, ymi ), (xma, yma), (255, 25, 55), 2)
    #cv2.imshow("box", frame)
    #cv2.waitKey(0)
    return frame, dimension_img[0], dimension_img[1], xmi, ymi, xma, yma

def main1():
    drawboundingbox2("../data/frame/charger/frame00213.jpg")


def get_all_steps(input_directory_of_obj):
  '''
  returns all assembly steps of the app (obj)
  '''
  steps = []
  for root, dirnames, filenames in os.walk(input_directory_of_obj):
    if len(dirnames) > 0: steps = dirnames
  print(steps)
  return steps
def main():

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")

    # Get all necessity paths
    labeling_directory = config_object["LABELING_DIRECTORY"]
    global OUTPUT_DIR
    OUTPUT_DIR = labeling_directory["annotated_frame_dir"]
    global INPUT_DIR
    INPUT_DIR = labeling_directory["frame_dir"]
    global CSV_DIR
    CSV_DIR = labeling_directory["csv_dir"]
    global LABEL_MAP_DIR
    LABEL_MAP_DIR = labeling_directory["label_map_dir"]

    global ABNORMAL_ANNOTATED_FRAME_DIR
    ABNORMAL_ANNOTATED_FRAME_DIR = labeling_directory["ABNORMAL_ANNOTATED_FRAME_DIR"]

    dataset_name = config_object["DATASET_NAME"]
    global obj_
    obj_ = dataset_name["obj"]
    steps = get_all_steps(os.path.join(INPUT_DIR, obj_))

    print("drawing bounding box for dataset ", obj_)

    for obj in steps:
        start_time = time.time()
        print("step ", obj)
        ##
        area_boundingbox = []
        list_boundingbox_frames = []
        ##
        #print("Labeling data for", obj)
        # Label map info
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        item = string_int_label_map_pb2.StringIntLabelMapItem()
        item.id = 1
        item.name = obj
        label_map.item.append(item)
        if not os.path.exists(os.path.join(LABEL_MAP_DIR, obj_, obj)):
            os.makedirs(os.path.join(LABEL_MAP_DIR, obj_, obj))
        with open(os.path.join(LABEL_MAP_DIR,obj_, obj, "label_map.pbtxt"), 'w') as f:
            f.write(text_format.MessageToString(label_map))

        output_folder = os.path.join(OUTPUT_DIR, obj_, obj)
        abnormal_annotated_frame_folder = os.path.join(ABNORMAL_ANNOTATED_FRAME_DIR,obj_, obj)
        create_folder_if_not_exist(abnormal_annotated_frame_folder)

        create_folder_if_not_exist(output_folder)

        csv_path = os.path.join(CSV_DIR, obj_)
        isExist = os.path.exists(csv_path)
        if not isExist:
            os.makedirs(csv_path)

        # create csv file to keep all information of a frame and its boundingbox
        csv_file = csv_path + "/" + obj + ".csv"
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print("Removed old csv file!")
        csv_file_path = Path(csv_file)
        csv_file_path.touch(exist_ok=True)
        csv_out = open(csv_file_path, 'w')
        header = ['filename', 'height', 'width', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.writer(csv_out)
        writer.writerow(header)

        # loop through all frames @ INPUT_DIR to draw boundingbox
        obj_img_files = os.listdir(os.path.join(INPUT_DIR,obj_, obj))
        obj_img_files.sort()
        #print("Drawing bounding box for frames in ", os.path.join(INPUT_DIR,obj_, obj), "....")
        for f in obj_img_files:
            # print(f)
            obj_img_file = os.path.join(INPUT_DIR,obj_, obj, f)
            result, height, width, xmin, ymin, xmax, ymax = drawboundingbox2(obj_img_file)
            #
            perimeter = ((xmax - xmin) + (ymax - ymin)) * 2
            area_boundingbox.append(perimeter)
            #
            tuple = (f, perimeter)
            list_boundingbox_frames.append(tuple)
            #
            row = [f, height, width, obj, xmin, ymin, xmax, ymax]
            # write info to csv file
            writer.writerow(row)
            # save the annotated frame for double-checking later
            #cv2.imwrite(os.path.join(output_folder, f), result)

        # check to remove abnormal boundingbox frames
        #remove_abnormal_frame(list_boundingbox_frames, area_boundingbox, output_folder,
        #                          abnormal_annotated_frame_folder)
        csv_out.close()
        print("Completed Labeling! All annotated frames are saved in", output_folder)
        print("--- %s seconds ---" % (time.time() - start_time))


def remove_abnormal_frame(list_boundingbox_frames, area_boundingbox, output_folder, abnormal_annotated_frame_folder):


    #mean_area = sum(area_boundingbox)/len(area_boundingbox)
    median_area = np.percentile(area_boundingbox, 50)
    q1_area = np.percentile(area_boundingbox, 25)
    q3_area = np.percentile(area_boundingbox, 75)
    iqr = q3_area - q1_area
    upper_fence = q3_area + 1.5*iqr
    lower_fence = q1_area - 1.5*iqr
    print("upper_fence", upper_fence)
    print("lower_fence", lower_fence)
    for i in list_boundingbox_frames:
        if i[1] > upper_fence or i[1] < lower_fence:
            #remove this frame because its area is greater than the mean area
            shutil.move(os.path.join(output_folder, i[0]), os.path.join(abnormal_annotated_frame_folder, i[0]))

def create_folder_if_not_exist(path_name):
    isExist = os.path.exists(path_name)
    if not isExist:
        os.makedirs(path_name)
    else:
        shutil.rmtree(path_name)
        print("Cleaning data in ", path_name, "....")
        os.makedirs(path_name)
if __name__ == '__main__':
    main()