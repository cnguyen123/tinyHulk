import argparse
import os
import pathlib

import tensorflow as tf
from pathlib import Path
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util

from google.protobuf import text_format
from configparser import ConfigParser
import glob

#######################3
#CHANH: IMPORTANCE NOTE: SET PARAMEMTER --combine-labels before running
###########################3
DEFAULT_CLASS_TEXT = 'default'
SINGLE_CLASS_NUM = 1

TFRECORD_NAME = 'merged.tfrecord'
LABEL_MAP_NAME = 'label_map.pbtxt'
INPUT_DATA_PATH='../data/tf_data/'
OUTPUT_MERGE_DATA_PATH='../data/merged_tf/'
obj = ""

IMAGE_FEATURE_DESCRIPTION = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}


def create_tf_example(parsed, class_text, class_num):
    height = parsed['image/height'].numpy()
    width = parsed['image/width'].numpy()
    filename = parsed['image/filename'].numpy()
    source_id = parsed['image/source_id'].numpy()
    encoded_image_data = parsed['image/encoded'].numpy()
    image_format = parsed['image/format'].numpy()
    xmins = [value.numpy() for value in parsed['image/object/bbox/xmin'].values]
    xmaxs = [value.numpy() for value in parsed['image/object/bbox/xmax'].values]
    ymins = [value.numpy() for value in parsed['image/object/bbox/ymin'].values]
    ymaxs = [value.numpy() for value in parsed['image/object/bbox/ymax'].values]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(source_id),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [class_text.encode('utf8')]),
        'image/object/class/label': dataset_util.int64_list_feature(
            [class_num]),
    }))
    return tf_example


def main():
    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")
    # Get all necessity paths
    dataset_name = config_object["DATASET_NAME"]
    global obj
    obj = dataset_name["obj"]

    TFDATA_directory = config_object["TFDATA_DIRECTORY"]

    global OUTPUT_MERGE_DATA_PATH
    OUTPUT_MERGE_DATA_PATH = os.path.join(TFDATA_directory["MERGED_TFDATA_DIR"], obj)

    global INPUT_DATA_PATH
    INPUT_DATA_PATH = os.path.join(TFDATA_directory["TFDATA_DIR"], obj)


    parser = argparse.ArgumentParser()
    parser.add_argument('--combine-labels', action='store_true')
    args = parser.parse_args()
    combine_labels = args.combine_labels

    label_map = string_int_label_map_pb2.StringIntLabelMap()


    isExist = os.path.exists(Path(OUTPUT_MERGE_DATA_PATH))

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(Path(OUTPUT_MERGE_DATA_PATH))
    writer = tf.io.TFRecordWriter(os.path.join(OUTPUT_MERGE_DATA_PATH,TFRECORD_NAME))

    class_nums = {}
    next_class_num = 1
    all_tf_files = []
    for path, subdirs, files in os.walk(INPUT_DATA_PATH):
        for name in files:
            #print(os.path.join(path, name))
            if pathlib.Path(name).suffix == ".tfrecord":
                all_tf_files.append(os.path.join(path, name))

    for full_dataset_path in all_tf_files:
        print(full_dataset_path)

        #full_dataset_path = os.path.join(dataset_dir, 'default.tfrecord')
        dataset = tf.data.TFRecordDataset(full_dataset_path)

        it = iter(dataset)
        for value in it:
            parsed = tf.io.parse_single_example(
                value, IMAGE_FEATURE_DESCRIPTION)
            num_values = len(parsed['image/object/class/text'].values)
            if num_values == 0:
                continue
            if num_values != 1:
                raise Exception


            if combine_labels:
                #print(combine_labels)
                class_text = DEFAULT_CLASS_TEXT
                class_num = SINGLE_CLASS_NUM
            else:

                class_text = parsed['image/object/class/text'].values.numpy()[0].decode(
                    'utf8')
                class_num = class_nums.get(class_text)

                if class_num is None:
                    class_nums[class_text] = next_class_num
                    class_num = next_class_num
                    next_class_num += 1
            #print(class_text)
            tf_example = create_tf_example(
                parsed, class_text, class_num)
            writer.write(tf_example.SerializeToString())

    writer.close()

    item = string_int_label_map_pb2.StringIntLabelMapItem()
    item.id = class_num
    item.name = DEFAULT_CLASS_TEXT
    label_map.item.append(item)
    with open(os.path.join(OUTPUT_MERGE_DATA_PATH,LABEL_MAP_NAME), 'w') as f:
        f.write(text_format.MessageToString(label_map))


if __name__ == '__main__':
    main()



