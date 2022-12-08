#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Chanh Nguyen
# Created Date: 02/02/22
# version ='1.0'
# ---------------------------------------------------------------------------
"""Generate tfrecord data for the annotated frames. \n
Note:
  - OUTPUT_DIR: where the annotated frames are saved
  - INPUT_DIR: where the frames locate
  - CSV_DIR: where the csv file (the file keeps information of frame and its boundingbox) locates
"""
# ---------------------------------------------------------------------------

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import re
import os
import io
import pandas as pd

import tensorflow as tf

from object_detection.utils import dataset_util


from PIL import Image

from collections import namedtuple

from configparser import ConfigParser



def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}
    prog = re.compile(r"  id: [0-9]\n")
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif prog.match(line): #"id" in line:

                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()
                item_name = item_name.strip('\"')

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items

def split(df, group):

    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map[row['class']])


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(b''),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    # if group.filename == "frame01132.jpg":
    #     # draw again annotation to test if it is correct or not
    #     where_to_save = os.path.join(NEW_ANNOTATION_DIR, "short_u_bar")
    #     fr = cv2.imread(os.path.join(FRAME_DIR, "short_u_bar", group.filename))
    #     xmi = int(xmins[0] * width)
    #     ymi = int(ymins[0] * height)
    #     yma = int(ymaxs[0] * height)
    #     xma = int(xmaxs[0] * width)
    #     cv2.rectangle(fr, (xmi, ymi), (xma, yma), (255, 25, 55), 2)
    #     cv2.imwrite(os.path.join(where_to_save, group.filename), fr)
    #     # end draw annotation



    return tf_example

#obj_ = ['step0', 'step1','step2', 'step3', 'step4', 'step5', 'step6', 'step7']
obj_ = ""

OUTPUT_DIR = ""
#FRAME_DIR = "../data/frame/diff_background/" #uncomment this line to create tfdata for diff_background frames
FRAME_DIR = ""
CSV_DIR = ""

LABEL_MAP_DIR = ""
ANNOTATED_FRAME_DIR = ""

#NEW_ANNOTATION_DIR = "../data/double_check_annotated_frame/"

def get_all_steps(input_directory_of_obj):
  '''
  returns all assembly steps of the app (obj)
  '''
  steps = []
  for root, dirnames, filenames in os.walk(input_directory_of_obj):
    if len(dirnames) > 0: steps = dirnames
  return steps

def main():

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")
    # Get all necessity paths
    TFDATA_directory = config_object["TFDATA_DIRECTORY"]

    global OUTPUT_DIR
    OUTPUT_DIR = TFDATA_directory["TFDATA_DIR"]

    global FRAME_DIR
    #FRAME_DIR = TFDATA_directory["FRAME_DIR"]
    FRAME_DIR = TFDATA_directory["FRAME_DIR_DIFF"]
    #FRAME_DIR = "../data/frame/diff_background2"
    global CSV_DIR
    CSV_DIR = TFDATA_directory["CSV_DIR"]

    global ANNOTATED_FRAME_DIR
    ANNOTATED_FRAME_DIR = TFDATA_directory["ANNOTATED_FRAME_DIR"]

    global LABEL_MAP_DIR
    LABEL_MAP_DIR = TFDATA_directory["LABEL_MAP_DIR"]

    dataset_name = config_object["DATASET_NAME"]
    global obj_
    obj_ = dataset_name["obj"]
    ##
    steps = get_all_steps(os.path.join(FRAME_DIR, obj_))
    steps = ['step3']
    print(steps)
    print("generating tfrecord for each step of dataset ", obj_)
    #subfolders = [f.name for f in os.scandir("../data/frame/diff_background/0wheel/") if f.is_dir()]
    for f in [""]:

        for obj in steps:
            print("step:", obj)
            LABEL_MAP_FILE = os.path.join(LABEL_MAP_DIR,obj_,  obj, "label_map.pbtxt" )
            #TF_RECORD_FILE_NAME = obj + ".tfrecord"
            TF_RECORD_FILE_NAME = obj + "_diff_2.tfrecord" #uncomment this line to create tfdata for diff_background frames
            label_map = read_label_map(LABEL_MAP_FILE)
            output_folder = os.path.join(OUTPUT_DIR,obj_, obj)

            # Check if the specified path exists
            isExist = os.path.exists(output_folder)
            if not isExist:
                os.makedirs(output_folder)

            output_path = os.path.join(output_folder, TF_RECORD_FILE_NAME)
            writer = tf.io.TFRecordWriter(output_path)
            path = os.path.join(FRAME_DIR, obj_, obj,f )  # path to frame folder back_ph_4
            csv_file = os.path.join(CSV_DIR, obj_) + "/" + obj + ".csv"
            examples = pd.read_csv(csv_file)
            grouped = split(examples, 'filename')
            #all_annotated_frame = os.listdir(os.path.join(ANNOTATED_FRAME_DIR, obj))
            for group in grouped:
                #if group.filename not in all_annotated_frame:
                #    continue
                tf_example = create_tf_example(group, path, label_map)
                writer.write(tf_example.SerializeToString())

            writer.close()
            print('Completed creating the TFRecords: {}'.format(output_path))





if __name__ == '__main__':


    #subfolders = [f.name for f in os.scandir("../data/frame/diff_background/0wheel/") if f.is_dir()]


    main()
