import cv2
import numpy as np
import os
from pathlib import Path
from configparser import ConfigParser

#def nothing(x):
#    pass
from matplotlib import pyplot as plt

INPUT_DIR = ""
OUTPUT_DIR = ""
BACKGROUND_IMGS = ""

ANNNOTATED_FRAME_DIR = ""

obj_ = ""
def change_background(bg_img_file):
    for obj in obj_:
        print("change background for ", obj)
        annotated_frame_list = os.listdir(os.path.join(ANNNOTATED_FRAME_DIR, obj))
        # First we need to estimate u_green, l_green by calling metric_tune(frame)
        input_frames_directory = os.path.join(INPUT_DIR, obj)
        obj_img_files = os.listdir(input_frames_directory)

        ## take only frames in annotated_frames directory for changing background
        obj_img_files = list(set(annotated_frame_list).intersection(obj_img_files))


        ff = cv2.imread(os.path.join(input_frames_directory, obj_img_files[0]))
        bg_img = cv2.imread(bg_img_file)
        u_green, l_green = metric_tune(ff, bg_img)
        print(u_green)

        #background_imgs = os.listdir(BACKGROUND_IMGS)
        #
        output_step_folder = os.path.join(OUTPUT_DIR, obj,Path(bg_img_file).stem)
        #

        print(" Changing background of frames in folder ", input_frames_directory, "...")

        # Check whether the specified path exists or not
        os.makedirs(output_step_folder, exist_ok=True)

        for f in obj_img_files:

            # print(f)
            obj_img_file = os.path.join(input_frames_directory, f)
            obj_img = cv2.imread(obj_img_file)
            new_frame = change_bg(obj_img, bg_img, u_green, l_green)

            cv2.imwrite(os.path.join(output_step_folder, f), new_frame)




def metric_tune(frame, bg_img):

    title_window = 'Color Tuned'
    cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
    ## [on_trackbar]
    def on_trackbar(value):
        #

        image = bg_img

        a = frame.shape

        image = cv2.resize(bg_img, (a[1], a[0]))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", title_window)
        l_s = cv2.getTrackbarPos("L - S", title_window)
        l_v = cv2.getTrackbarPos("L - V", title_window)
        u_h = cv2.getTrackbarPos("U - H", title_window)
        u_s = cv2.getTrackbarPos("U - S", title_window)
        u_v = cv2.getTrackbarPos("U - V", title_window)
        u_green = np.array([u_h, u_s, u_v])
        l_green = np.array([l_h, l_s, l_v])

        mask = cv2.inRange(hsv, l_green, u_green)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        f = frame - res
        green_screen = np.where(f == 0, image, f)

        #
        cv2.imshow(title_window, green_screen)


    cv2.namedWindow(title_window)

    ## [create_trackbar]
    cv2.createTrackbar("L - H", title_window, 0, 179, on_trackbar)
    cv2.createTrackbar("L - S", title_window, 0, 255, on_trackbar)
    cv2.createTrackbar("L - V", title_window, 0, 255, on_trackbar)
    cv2.createTrackbar("U - H", title_window, 179, 179, on_trackbar)
    cv2.createTrackbar("U - S", title_window, 255, 255, on_trackbar)
    cv2.createTrackbar("U - V", title_window, 255, 255, on_trackbar)
    #cv2.resizeWindow(title_window, 100, 100)

    cv2.waitKey(0)

    l_h = cv2.getTrackbarPos("L - H", title_window)
    l_s = cv2.getTrackbarPos("L - S", title_window)
    l_v = cv2.getTrackbarPos("L - V", title_window)
    u_h = cv2.getTrackbarPos("U - H", title_window)
    u_s = cv2.getTrackbarPos("U - S", title_window)
    u_v = cv2.getTrackbarPos("U - V", title_window)
    u_green = np.array([u_h, u_s, u_v])
    l_green = np.array([l_h, l_s, l_v])
    cv2.destroyAllWindows()
    return (u_green, l_green)

def test():
    frame = "../data/frame/transform/step1/frame00002.jpg"
    global BACKGROUND_IMGS
    BACKGROUND_IMGS = "../data/background_img/"
    ff = cv2.imread(frame)
    metric_tune(ff)
def change_bg(frame, bg_img, u_green, l_green):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    f_shape = frame.shape
    image = cv2.resize(bg_img, (f_shape[1], f_shape[0]))


    mask = cv2.inRange(hsv, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame - res
    green_screen = np.where(f == 0, image, f)
    return green_screen


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
    OUTPUT_DIR = labeling_directory["BACKGROUND_DIFF_DIR"]

    global BACKGROUND_IMGS
    BACKGROUND_IMGS = labeling_directory["BACKGROUND_IMG_DIR"]
    global  ANNNOTATED_FRAME_DIR
    ANNNOTATED_FRAME_DIR = labeling_directory["ANNOTATED_FRAME_DIR"]

    background_imgs = os.listdir(BACKGROUND_IMGS)

    for bg_img_file in background_imgs:
        change_background(os.path.join(BACKGROUND_IMGS, bg_img_file))

    #change_background()


if __name__ == '__main__':
    main()
