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


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def getinfo(group):
    image = cv2.imread(os.path.join(INPUT_DIR, group.filename))
    width, height, _ = image.shape


    xmins = 0
    xmaxs = 0
    ymins = 0
    ymaxs = 0

    for index, row in group.object.iterrows():
        xmins= int(row['xmin'] )
        xmaxs = int(row['xmax'])
        ymins = int (row['ymin'] )
        ymaxs = int(row['ymax'] )
    return xmins, ymins, xmaxs, ymaxs
INPUT_DIR = "../data/frame/diff_background/0wheel"
obj = "0wheel"
OUTPUTDIR = "../data/test/"
CSV_DIR = "../data/csv_data/"
csv_file = os.path.join(CSV_DIR) + "/" + obj + ".csv"
examples = pd.read_csv(csv_file)
grouped = split(examples, 'filename')

for group in grouped:

    img = cv2.imread(os.path.join(INPUT_DIR, group.filename))
    xmi, ymi, xma, yma = getinfo(group)
    cv2.rectangle(img, (xmi, ymi), (xma, yma), (255, 25, 55), 2)
    cv2.imwrite(os.path.join(OUTPUTDIR, group.filename), img)

