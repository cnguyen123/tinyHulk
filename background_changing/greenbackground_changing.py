import cv2
import numpy as np
import os
import random
from configparser import ConfigParser

#def nothing(x):
#    pass
from matplotlib import pyplot as plt

INPUT_DIR = ""
OUTPUT_DIR = ""
BACKGROUND_IMGS = ""

ANNNOTATED_FRAME_DIR = ""
#obj_ = ['step0', 'step1','step2', 'step3', 'step4', 'step5', 'step6', 'step7']
obj_ = ""
def get_all_steps(input_directory_of_obj):
  '''
  returns all assembly steps of the app (obj)
  '''
  steps = []
  for root, dirnames, filenames in os.walk(input_directory_of_obj):
    if len(dirnames) > 0: steps = dirnames
  print(steps)
  return steps
def change_background():
    steps= get_all_steps(os.path.join(INPUT_DIR, obj_))

    for step in steps:
        print("change background for ", step)
        #annotated_frame_list = os.listdir(os.path.join(ANNNOTATED_FRAME_DIR, step))
        # First we need to estimate u_green, l_green by calling metric_tune(frame)
        input_frames_directory = os.path.join(INPUT_DIR,obj_, step)
        obj_img_files = os.listdir(input_frames_directory)

        ## take only frames in annotated_frames directory for changing background
        #obj_img_files = list(set(annotated_frame_list).intersection(obj_img_files))


        ff = cv2.imread(os.path.join(input_frames_directory, obj_img_files[0]))

        u_green, l_green = metric_tune(ff)
        print(u_green)

        background_imgs = os.listdir(BACKGROUND_IMGS)
        #
        output_step_folder = os.path.join(OUTPUT_DIR, obj_, step)
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
            new_frame = change_bg(obj_img, bg_img, u_green, l_green)

            cv2.imwrite(os.path.join(output_step_folder, f), new_frame)




def metric_tune(frame):

    background_imgs = os.listdir(BACKGROUND_IMGS)
    bg_img_files = os.path.join(BACKGROUND_IMGS, random.choice(background_imgs))
    bg_img = cv2.imread(bg_img_files)
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

def change_bg_new(frame = "../data/frame/random/frame00000.jpg", bg_img = "../data/background_img/brick.jpg" ):
    frame = cv2.imread(frame)
    #metric_tune(frame)
    marvel_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # defining range to excloude the green color from the image
    # the list contain -> [ value of the Red, value of the Green, value of the Blue
    lower_range = np.array([5, 40, 5])
    upper_range = np.array([78, 255, 150])

    # form [ (0 ->110) for Red, (100 -> 255) for Green, ...]
    mask = cv2.inRange(marvel_image, lower_range, upper_range)
    # set all other areas to zero except where mask area
    marvel_image[mask != 0] = [0, 0, 0]
    background_Image = cv2.imread(bg_img)
    background_Image = cv2.cvtColor(background_Image, cv2.COLOR_BGR2RGB)

    # Note: the baground image may not as the same size as frist image
    # so we run cv2.resize to ensure that those image are in the same size
    background_Image = cv2.resize(background_Image, (frame.shape[1], frame.shape[0]))
    #set the mask area with black to be replaced with Thor Image
    background_Image[mask == 0] = [0, 0, 0]
    cv2.imshow("object image",marvel_image)
    cv2.imshow("background", background_Image)
    complete_image = background_Image + marvel_image
    cv2.imshow("result", complete_image)
    cv2.waitKey(0)

def main():
    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("../config.ini")
    # Get all necessity paths
    dataset_name = config_object["DATASET_NAME"]
    global obj_
    obj_ = dataset_name["obj"]

    labeling_directory = config_object["LABELING_DIRECTORY"]
    global INPUT_DIR
    INPUT_DIR = labeling_directory["FRAME_DIR"]

    global OUTPUT_DIR
    OUTPUT_DIR = labeling_directory["BACKGROUND_DIFF_DIR"]

    global BACKGROUND_IMGS
    BACKGROUND_IMGS = labeling_directory["BACKGROUND_IMG_DIR"]
    global  ANNNOTATED_FRAME_DIR
    ANNNOTATED_FRAME_DIR = labeling_directory["ANNOTATED_FRAME_DIR"]
    change_background()


if __name__ == '__main__':
    main()