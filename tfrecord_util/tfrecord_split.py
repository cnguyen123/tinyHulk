import tensorflow as tf
import os
from configparser import ConfigParser
from pathlib import Path
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

TFRECORD_NAME = 'merged.tfrecord'
TFRECORD_NAME = 'green_only.tfrecord'
TRAIN_TFRECORD_NAME = 'train.tfrecord'
TEST_TFRECORD_NAME = 'test.tfrecord'
VAL_TFRECORD_NAME = 'val.tfrecord'


INPUT_DATA_PATH=''
OUTPUT_SPLITTED_DATA_PATH=''
obj = ""
def main():

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")
    # Get all necessity paths
    dataset_name = config_object["DATASET_NAME"]
    global obj
    obj = dataset_name["obj"]

    TFDATA_directory = config_object["TFDATA_DIRECTORY"]

    global INPUT_DATA_PATH
    INPUT_DATA_PATH = os.path.join(TFDATA_directory["MERGED_TFDATA_DIR"], obj)

    global OUTPUT_SPLITTED_DATA_PATH
    OUTPUT_SPLITTED_DATA_PATH = os.path.join(TFDATA_directory["SPLITTED_TFDATA_DIR"], obj)
    INPUT_DATA_PATH = "../data/"
    OUTPUT_SPLITTED_DATA_PATH = "../data/"
    split_paras = config_object["SPLIT_PARAMETER"]

    #percentage of dataset in each train, val, test set
    train_split = float(split_paras["TRAIN_SPLIT"])
    val_split = float(split_paras["VAL_SPLIT"])
    test_split = float(split_paras["TEST_SPLIT"])

    # merged_tfrecord
    ds = tf.data.TFRecordDataset(
    os.path.join(INPUT_DATA_PATH,TFRECORD_NAME), compression_type=None, buffer_size=None, num_parallel_reads=None)
    # merged_tfrecord size
    ds_size = sum(1 for record in ds)

    isExist = os.path.exists(OUTPUT_SPLITTED_DATA_PATH)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(OUTPUT_SPLITTED_DATA_PATH)

    train_set, val_set, test_set = get_dataset_partitions_tf(ds, ds_size, train_split, val_split, test_split )

    train_writer = tf.data.experimental.TFRecordWriter(os.path.join(OUTPUT_SPLITTED_DATA_PATH, TRAIN_TFRECORD_NAME ))
    train_writer.write(train_set)

    test_writer = tf.data.experimental.TFRecordWriter(os.path.join(OUTPUT_SPLITTED_DATA_PATH, TEST_TFRECORD_NAME))
    test_writer.write(test_set)

    val_writer = tf.data.experimental.TFRecordWriter(os.path.join(OUTPUT_SPLITTED_DATA_PATH, VAL_TFRECORD_NAME))
    val_writer.write(val_set)



if __name__ == '__main__':
    main()
