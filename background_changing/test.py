import math


import csv

import cv2

import os
import random
from configparser import ConfigParser





CSV_FILE = "../data/csv_data/0wheel.csv"
def load_boundingbox(image_name):
    #Read csv
    file = open(CSV_FILE, 'r')
    ct = 0
    for row in csv.reader(file):
        if row[ct] == image_name:
            #print(row[1])
            return row[4:8]

def detect_objects(frame):
    # Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("mask", mask)
    objects_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:

            objects_contours.append(cnt)

    return objects_contours




INPUT_DIR = ""
OUTPUT_DIR = ""
BACKGROUND_IMGS = ""

ANNNOTATED_FRAME_DIR = ""
#obj_ = ['step0', 'step1','step2', 'step3', 'step4', 'step5', 'step6', 'step7']
obj_ = ""
def change_background():
    for obj in obj_:
        print("change background for ", obj)

        # First we need to estimate u_green, l_green by calling metric_tune(frame)
        input_frames_directory = os.path.join(INPUT_DIR, obj)
        obj_img_files = os.listdir(input_frames_directory)

        ## take only frames in annotated_frames directory for changing background
        #obj_img_files = list(set(annotated_frame_list).intersection(obj_img_files))


        ff = cv2.imread(os.path.join(input_frames_directory, obj_img_files[0]))



        background_imgs = os.listdir(BACKGROUND_IMGS)
        #
        output_step_folder = os.path.join(OUTPUT_DIR, obj)
        #

        print(" Changing background of frames in folder ", input_frames_directory, "...")

        # Check whether the specified path exists or not
        os.makedirs(output_step_folder, exist_ok=True)

        for f in obj_img_files:
            bg_img_file = os.path.join(BACKGROUND_IMGS, random.choice(background_imgs))
            bg_img = cv2.imread(bg_img_file)
            # print(f)
            obj_img_file = os.path.join(input_frames_directory, f)
            obj_img = cv2.imread(obj_img_file)
            new_frame = change_bg(obj_img, bg_img, f)

            cv2.imwrite(os.path.join(output_step_folder, f), new_frame)





def change_bg(img, bg_img, imge_name):
    #imge_name = "frame01777.jpg"
    #img = cv2.imread(os.path.join("../data/frame/0wheel/", imge_name))
    bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

    [xmi, ymi, xma, yma] = load_boundingbox(imge_name)
    xmi = math.ceil(float(xmi))
    ymi = math.ceil(float(ymi))
    xma = math.ceil(float(xma))
    yma = math.ceil(float(yma))
    #cv2.imshow("image_original", img)

    cropped_image = img[ymi:yma, xmi:xma]

    #cv2.imshow("cropped", cropped_image)
    #cv2.waitKey(0)

    o_contours = detect_objects(cropped_image)
    # a = cv2.drawContours(cropped_image, [c for c in o_contours], -1, (0, 0, 255), 2)
    x_i = 100
    y_i = 100
    x_m = -100
    y_m = -100
    for cnt in o_contours:
        # print(cnt.shape)
        # Get rect
        rect = cv2.minAreaRect(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # (x, y), (w, h), angle = rect
        if x + w > x_m:
            x_m = x + w
        if x < x_i:
            x_i = x
        if y + h > y_m:
            y_m = y + h
        if y < y_i:
            y_i = y

        # Display rectangle
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #print(box)
        # cv2.polylines(cropped_image, [box], True, (255, 0, 0), 5)

    # cv2.rectangle(cropped_image, (int(x_i), int(y_i)), (int(x_m), int(y_m)),  (255, 0, 0), 2)
    # cv2.rectangle(frame, (int(x), int(y)), (int(x+h), int(y+w)), (0, 255, 255), 2)
    # frame = cv2.resize(img, (640,640))
    # cv2.rectangle(img, (int(x_i + xmi), int(y_i + ymi)), (int(x_m + xmi), int(y_m + ymi)),  (255, 255, 0), 2)

    #ROI = cropped_image[y_i:y_m, x_i:x_m]
    y_i_bg = y_i + ymi
    y_m_bg = y_m + ymi
    x_i_bg = x_i + xmi
    x_m_bg = x_m + xmi

    bg_img[y_i_bg:y_m_bg, x_i_bg:x_m_bg] = cropped_image[y_i:y_m, x_i:x_m]

    return bg_img


def main():
    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")
    # Get all necessity paths
    dataset_name = config_object["DATASET_NAME"]
    global obj_
    obj_ = [dataset_name["obj"]]

    labeling_directory = config_object["LABELING_DIRECTORY"]
    global INPUT_DIR
    INPUT_DIR = labeling_directory["FRAME_DIR"]

    global OUTPUT_DIR
    #OUTPUT_DIR = labeling_directory["BACKGROUND_DIFF_DIR"]
    OUTPUT_DIR = "../data/frame/diff_background2"
    global BACKGROUND_IMGS
    BACKGROUND_IMGS = labeling_directory["BACKGROUND_IMG_DIR"]

    change_background()


if __name__ == '__main__':
    main()