import csv
import os.path
import shutil
from pathlib import Path
import albumentations as A
import random
import numpy as np
import pandas as pd
import cv2
from collections import namedtuple

from configparser import ConfigParser
from matplotlib import pyplot as plt

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)



FRAME_DIR = ""
ANNOTATED_FRAME_DIR = ""
CSV_DIR =""
OUTPUT_ANNOTATED_FRAME_TRANSFORM_DIR=""
OUTPUT_TRANSFORMED_FRAME_DIR = ""
obj_ = ""
def main():

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")

    # Get all necessity paths
    labeling_directory = config_object["LABELING_DIRECTORY"]
    global ANNOTATED_FRAME_DIR
    ANNOTATED_FRAME_DIR = labeling_directory["annotated_frame_dir"]
    global FRAME_DIR
    FRAME_DIR = labeling_directory["frame_dir"]
    global CSV_DIR
    CSV_DIR = labeling_directory["csv_dir"]

    transform_directory = config_object["TRANSFORM_DIRECTORY"]
    global OUTPUT_ANNOTATED_FRAME_TRANSFORM_DIR
    OUTPUT_ANNOTATED_FRAME_TRANSFORM_DIR = transform_directory["TRANSFORM_ANNOTATED_FRAME_DIR"]

    global OUTPUT_TRANSFORMED_FRAME_DIR
    OUTPUT_TRANSFORMED_FRAME_DIR = transform_directory["TRANSFORMED_FRAME_DIR"]


    dataset_name = config_object["DATASET_NAME"]
    global obj_
    obj_ = [dataset_name["obj"]]


    for obj in obj_:

        ##
        area_boundingbox = []
        list_boundingbox_frames = []
        ##

        frame_path = os.path.join(FRAME_DIR, obj)
        frame_files = os.listdir(frame_path)
        create_folder_if_not_exist(os.path.join(OUTPUT_ANNOTATED_FRAME_TRANSFORM_DIR, obj))
        transformed_frame_dir = os.path.join(OUTPUT_TRANSFORMED_FRAME_DIR, obj)
        create_folder_if_not_exist(transformed_frame_dir)
        annotated_frame_path = os.path.join(ANNOTATED_FRAME_DIR, obj)
        annotated_frame_files = os.listdir(annotated_frame_path)
        path = os.path.join(FRAME_DIR, obj)
        transform_frame_files = list(set(annotated_frame_files).intersection(frame_files))


        # csv file of new transformed frames
        isExist = os.path.exists(CSV_DIR)
        if not isExist:
            os.makedirs(CSV_DIR)

        # create csv file to keep all information of a frame and its boundingbox
        csv_tf_frame_file = CSV_DIR + "/" + obj + "_transformed.csv"
        if os.path.exists(csv_tf_frame_file):
            os.remove(csv_tf_frame_file)
            print("Removed old csv file:", csv_tf_frame_file)
        csv_file_path = Path(csv_tf_frame_file)
        csv_file_path.touch(exist_ok=True)
        csv_out = open(csv_file_path, 'w')
        header = ['filename', 'height', 'width', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.writer(csv_out)
        writer.writerow(header)
        print("CSV file keeps all information (labeling) of the new transformed frames is created at ", csv_tf_frame_file)
        ##


        #csv file of annotated green background frames
        csv_annotated_frame_file = os.path.join(CSV_DIR) + "/" + obj + ".csv"
        examples = pd.read_csv(csv_annotated_frame_file)
        grouped = split(examples, 'filename')

        print("transforming frames from ", frame_path, ". New transformed frames saved at the folder", transformed_frame_dir, "...")
        for group in grouped:
            if group.filename not in transform_frame_files:
                continue
            tf_frame, x_min, y_min, x_max, y_max = transform_frame(group, path)
            if(x_min != -1 and y_min!= -1 and x_max!=-1 and y_max!= -1):
                cv2.imwrite(os.path.join(transformed_frame_dir, group.filename), tf_frame)
                #
                perimeter = ((x_max - x_min) + (y_max - y_min)) * 2
                area_boundingbox.append(perimeter)
                #
                tuple = (group.filename, perimeter)
                list_boundingbox_frames.append(tuple)
                row = [group.filename, tf_frame.shape[1], tf_frame.shape[0], obj, x_min, y_min, x_max, y_max]
                # write info to csv file
                writer.writerow(row)
        remove_abnormal_frame(list_boundingbox_frames, area_boundingbox, transformed_frame_dir, "../data/abnormal_annotated_frame/step1/")



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]




def transform_frame(group, path):
    image = cv2.imread(os.path.join(path, group.filename))

    xmin = -1
    ymin = -1
    xmax = -1
    ymax = -1
    for index, row in group.object.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
    bboxes = [[xmin, ymin, xmax -xmin, ymax - ymin]]
    category_ids = [1]
    category_id_to_name = {1: 'bar'}


    transform = A.Compose([
        #A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.8),
        #A.ShiftScaleRotate(rotate_limit= (0, 90), p=1),
        A.Rotate(limit=90, always_apply=True, p=1,  interpolation=1, border_mode= 1),
        A.RandomBrightnessContrast(p=0.8, always_apply=True),
        A.ColorJitter(brightness=(0.5, 1.1), contrast=0.5, saturation=0.5, hue=0.3, always_apply=True, p=0.8),
        #A.RandomScale(scale_limit=(0.1, 0.8),  always_apply=True, p=0.9),
        #A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3)
    ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.8),
        # min_area is a value in pixels. If the area of a bounding box after augmentation becomes smaller than min_area,
        # Albumentations will drop that box. So the returned list of augmented bounding boxes won't contain that bounding box.
        #
        # min_visibility is a value between 0 and 1.
        # If the ratio of the bounding box area after augmentation to the area of the bounding box before augmentation
        # becomes smaller than min_visibility, Albumentations will drop that box.
        # So if the augmentation process cuts the most of the bounding box,
        # that box won't be present in the returned list of the augmented bounding boxes.
    )
    #random.seed(7)
    transform2 = A.Compose([
        #A.HorizontalFlip(p=0.8),

        A.Rotate(limit=(90,120), interpolation=1, border_mode=1, method='ellipse', always_apply=True, p=1),
        #A.RandomBrightnessContrast(p=0.6, always_apply=True)

    ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.8)

    )
    transformed = transform2(image=image, bboxes=bboxes, category_ids=category_ids)

    transformed_frame = transformed['image']
    transformed_frame_bbox = transformed['bboxes']
    if len(transformed_frame_bbox)>0:
        bbox = transformed_frame_bbox[0]
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        #cv2.rectangle(transformed_frame, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        return transformed_frame, x_min, y_min, x_max, y_max
    else:
        return transformed_frame, -1, -1, -1, -1



def create_folder_if_not_exist(path_name):
    isExist = os.path.exists(path_name)
    if not isExist:
        os.makedirs(path_name)
    else:
        shutil.rmtree(path_name)
        print("Cleaning data in ", path_name, "....")
        os.makedirs(path_name)



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

def test():
    img = cv2.imread("../data/frame/transform/step1/frame02073.jpg")
    #img = cv2.imread("../data/frame/step1/frame01382.jpg")
    xmin = 703#int(0.40052083 * img.shape[1])
    xmax = 955#int(0.53489584 * img.shape[1])
    ymin = 434#int(0.0 * img.shape[0])
    ymax = 703#int(0.18055555*img.shape[0])
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    img = cv2.resize(img, (1000, 1000))
    cv2.imshow("img", img)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
