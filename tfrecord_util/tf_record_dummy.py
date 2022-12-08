import cv2

import csv
from pathlib import Path


from configparser import ConfigParser
import os
import io
import pandas as pd

import tensorflow as tf


from object_detection.utils import dataset_util


from PIL import Image

from collections import namedtuple

OUTPUT_CSV_DIR = ""
INPUT_FRAME_DIR = ""
DUMMY_TFDATA_PATH = ""



def create_csv(obj="aruco"):

    # Check whether the specified path exists or not
    isExist = os.path.exists(OUTPUT_CSV_DIR)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(OUTPUT_CSV_DIR)


    obj_img_files = os.listdir(os.path.join(INPUT_FRAME_DIR, obj))
    csv_file =  obj + ".csv"
    csv_file_path = Path(os.path.join(OUTPUT_CSV_DIR, csv_file))
    print(csv_file_path)
    csv_file_path.touch(exist_ok=True)
    csv_out = open(csv_file_path, 'w')
    header = ['filename', 'height', 'width', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    writer = csv.writer(csv_out)
    writer.writerow(header)
    for f in obj_img_files:
        #print(f)
        # obj_img = os.path.join(INPUT_DIR, "object",f)
        obj_img = os.path.join(INPUT_FRAME_DIR, obj, f)
        img2 = cv2.imread(obj_img)
        dimension_image = img2.shape

        height, width, xmin, ymin, xmax, ymax = [dimension_image[0], dimension_image[1], -1, -1, -1, -1]

        row = [f, height, width, 'None', xmin, ymin, xmax, ymax]
        writer.writerow(row)


    csv_out.close()

def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group, path):
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
        classes.append(class_text_to_int(row['class']))

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
    return tf_example

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'None':
        return 1
    else:
        return None


def create_tf_record(obj = "phone"):


    output_path = DUMMY_TFDATA_PATH + obj + ".tfrecord"
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(INPUT_FRAME_DIR, obj)  # path to frame folder



    examples = pd.read_csv(OUTPUT_CSV_DIR + obj + ".csv")
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    # output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    # output_path = "../tf_data/chassis_1.tfrecord"
    print('Successfully created the TFRecords: {}'.format(output_path))
def main():




    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")

    # Get all necessity paths
    labeling_directory = config_object["LABELING_DIRECTORY"]

    global INPUT_FRAME_DIR
    INPUT_FRAME_DIR = labeling_directory["frame_dir"]
    global OUTPUT_CSV_DIR
    OUTPUT_CSV_DIR = labeling_directory["csv_dir"]

    dummy_directory = config_object["DUMMY_DIRECTORY"]
    global DUMMY_TFDATA_PATH
    DUMMY_TFDATA_PATH = dummy_directory["DUMMY_TFDATA_PATH"]
    dataset_name = config_object["DATASET_NAME"]
    global obj
    obj = dataset_name["obj"]
    create_csv(obj)
    create_tf_record(obj)

if __name__ == '__main__':
    main()






