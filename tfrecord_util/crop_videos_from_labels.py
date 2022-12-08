import os
import pathlib
from pathlib import Path
import tensorflow as tf

from PIL import Image

import glob


obj_ = ['full']
OUTPUT_DIR = "../data/cropped_image_data/"
INPUT_DIR = "../data/tf_data/input"

IMAGE_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}


def write_cropped_image(parsed, image_count):
    class_label = parsed['image/object/class/text'].values.numpy()[0].decode(
        'utf-8')
    tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = Image.fromarray(tf_image.numpy())

    x1 = parsed['image/object/bbox/xmin'].values[0].numpy() * image.width
    x2 = parsed['image/object/bbox/xmax'].values[0].numpy() * image.width
    y1 = parsed['image/object/bbox/ymin'].values[0].numpy() * image.height
    y2 = parsed['image/object/bbox/ymax'].values[0].numpy() * image.height

    cropped = image.crop((x1, y1, x2, y2))
    write_dir = os.path.join(OUTPUT_DIR, class_label)
    os.makedirs(write_dir, exist_ok=True)
    cropped.save(os.path.join(write_dir, '{}.jpg'.format(image_count)))


def main():
    for obj in obj_:
        print("Crop the object in bounding box location for", obj)
        image_count = 0
        input_dir = os.path.join(INPUT_DIR, obj)
        print(input_dir)
        for dataset_dir in glob.glob(os.path.join(input_dir, '*')):

            if pathlib.Path(dataset_dir).suffix != ".tfrecord":
                continue

            print(dataset_dir)
            full_dataset_path = dataset_dir
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

                write_cropped_image(parsed, image_count)
                image_count += 1



if __name__ == '__main__':
    main()
