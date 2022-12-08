# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
FRAME_PATH = "../data/frame"
obj = "0wheel"
IMAGES_PATH = os.path.sep.join([FRAME_PATH, obj])

CSV_PATH = "../data/csv_data"
ANNOTS_PATH = os.path.sep.join([CSV_PATH, "{}.csv".format(obj)])

# define the path to the base output directory
BASE_OUTPUT = "trained_model"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32