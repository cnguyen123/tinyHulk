from skimage.metrics import structural_similarity
import cv2
import numpy as np
import argparse
import os
import glob
import csv
from pathlib import Path
OUTPUT_DIR = "../data_turntable/annotated_frame/"
INPUT_DIR = "../data_turntable/frame/"
CSV_DIR = "../data_turntable/csv_data/"
BACKGROUND_DIR = "../data_turntable/background/"

ANNOTATED_DIR = "../data_turntable/annotated_frame"
FRAME_DIR = "../data_turntable/frame/"
CSV_DIR = "../data_turntable/csv_data/"
BACKGROUND_DIR = "../data_turntable/background/"
VIDEO_INPUT_DIR = "../data_turntable/video"
obj = "0wheel1"

def drawboundingbox(background_img, obj_img,padding = 10):
    bg = cv2.imread(background_img)

    obj = cv2.imread(obj_img)
    result = obj.copy()
    before = cv2.GaussianBlur(bg, (11, 11), 0)
    after = cv2.GaussianBlur(obj, (11, 11), 0)
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    #print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    max_area = -1
    x_draw = 0
    y_draw = 0
    w_draw = 0
    h_draw = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            x, y, w, h = cv2.boundingRect(c)
            if h * w > max_area:
                # second_area = max_area
                # x_draw2 = x_draw
                # y_draw2 = y_draw
                # w_draw2 = w_draw
                # h_draw2 = h_draw
                max_area = h * w
                x_draw = x
                y_draw = y
                w_draw = w
                h_draw = h
            #cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            #cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            #cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            #cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
    x_low = x_draw - padding
    y_low = y_draw - padding
    x_high = x_draw + w_draw + padding
    y_high = y_draw + h_draw + padding
    dimension_img = result.shape
    xmi = min(max(x_low , 0), dimension_img[1])
    ymi = min(max(y_low, 0), dimension_img[0])
    xma = max(min(x_high, dimension_img[1]), 0)
    yma = max(min(y_high, dimension_img[0]), 0)
    cv2.rectangle(result, (xmi, ymi), (xma, yma), (36, 255, 12), 2)
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey(0)


    return result, dimension_img[0], dimension_img[1], xmi, ymi, xma, yma
    #return result

def main():
    #video_to_frames()
    # obj_ = ["tray", "tray_1", "tray_2"]
    step = ""
    subtask = ["0wheel2", "0wheel1", "0wheel"]
    camera = [""]
    output_step_folder= os.path.join(OUTPUT_DIR, step)
    for task in subtask:
        for c in camera:
            print("....labelling for sub task {} with frames from camera {}...".format(task, c))
            output_folder = os.path.join(output_step_folder, c, task)
            # Check whether the specified path exists or not
            os.makedirs(output_folder, exist_ok=True)


            background_img_file = os.path.join(BACKGROUND_DIR,step, c) + "/" + task + ".jpg"
            background_img = cv2.imread(background_img_file)
            input_frames_directory = os.path.join(INPUT_DIR,step,  c, task)
            obj_img_files = os.listdir(input_frames_directory)
            csv_folder = os.path.join(CSV_DIR, step, c)
            os.makedirs(csv_folder, exist_ok=True)
            csv_file = csv_folder + "/" + task + ".csv"

            csv_file_path = Path(csv_file)
            csv_file_path.touch(exist_ok=True)
            csv_out = open(csv_file_path, 'w')
            header = ['filename', 'height', 'width', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            writer = csv.writer(csv_out)
            writer.writerow(header)
            for f in obj_img_files:
                #print(f)

                #test(os.path.join(input_frames_directory, f), background_img_file)
                obj_img_file = os.path.join(input_frames_directory, f)
                result, height, width, xmin, ymin, xmax, ymax = drawboundingbox(background_img_file, obj_img_file)
                row = [f, height, width, task, xmin, ymin, xmax, ymax]
                writer.writerow(row)
                cv2.imwrite(os.path.join(output_folder, f), result)

            csv_out.close()




def test(file_frame,background_img, padding = 10 ):
    bg = cv2.imread(background_img)

    obj = cv2.imread(file_frame)
    result = obj.copy()
    before = cv2.GaussianBlur(bg, (11, 11), 0)
    after = cv2.GaussianBlur(obj, (11, 11), 0)
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    # print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #mask = np.zeros(before.shape, dtype='uint8')
    #filled_after = after.copy()
    max_area = -1
    x_draw = 0
    y_draw = 0
    w_draw = 0
    h_draw = 0

    X_MIN = 10000
    Y_MIN = 10000
    X_MAX = -1000
    Y_MAX = -1000
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            if X_MIN > x_min:
                X_MIN = x_min
            if Y_MIN > y_min:
                Y_MIN = y_min
            if X_MAX < x_max:
                X_MAX = x_max
            if Y_MAX < y_max:
                Y_MAX = y_max
            if h * w > max_area:
                # second_area = max_area
                # x_draw2 = x_draw
                # y_draw2 = y_draw
                # w_draw2 = w_draw
                # h_draw2 = h_draw
                max_area = h * w
                x_draw = x
                y_draw = y
                w_draw = w
                h_draw = h
            cv2.rectangle(result, (x, y), (x + w, y + h), (36, 255, 12), 2)

            # cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            # cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
    x_low = x_draw - padding
    y_low = y_draw - padding
    x_high = x_draw + w_draw + padding
    y_high = y_draw + h_draw + padding
    cv2.rectangle(result, (x_low, y_low), (x_high, y_high), (36, 255, 12), 2)
    # cv2.imshow('before', before)
    # cv2.imshow('after', after)
    # cv2.imshow('diff', diff)
    # cv2.imshow('mask', mask)
    # cv2.imshow('filled after', filled_after)
    # cv2.waitKey(0)
    cv2.rectangle(result, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (255, 25, 55), 2)
    cv2.imshow("result", result)
    cv2.waitKey(0)


def video_to_frames():
  output_folder = os.path.join(FRAME_DIR, obj)

  # Check if the specified path exists
  isExist = os.path.exists(output_folder)
  if not isExist:
    os.makedirs(output_folder)

  videos_list = os.listdir(os.path.join(VIDEO_INPUT_DIR, obj))
  count = 0
  for v in videos_list:
    print("parsing frames from video",v,"...")
    video_file = os.path.join(VIDEO_INPUT_DIR, obj, v)
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()

    while success:
      name = "frame%05d.jpg" % count
      file_name = os.path.join(output_folder, name)
      cv2.imwrite(file_name, image)  # save frame as JPG file
      success, image = vidcap.read()
      #print('Read a new frame: ', success)
      count += 1

  print("Completed! All frames are saved at ", output_folder)

if __name__ == '__main__':
    main()