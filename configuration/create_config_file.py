import os
import configparser

configfile_name = "../config.ini"

# Check if there is already a configurtion file
if not os.path.isfile(configfile_name):
    # Create the configuration file as it doesn't exist yet
    cfgfile = open(configfile_name, "w")

    # Add content to the file
    Config = configparser.ConfigParser()
    Config.add_section("LABELING_DIRECTORY")
    Config.set("LABELING_DIRECTORY", "VIDEO_INPUT_DIR", "../data/video/")
    Config.set("LABELING_DIRECTORY", "FRAME_DIR", "../data/frame/")
    Config.set("LABELING_DIRECTORY", "ANNOTATED_FRAME_DIR", "../data/annotated_frame/")
    Config.set("LABELING_DIRECTORY", "CSV_DIR", "../data/csv_data/")
    Config.set("LABELING_DIRECTORY", "BACKGROUND_IMG_DIR", "../data/background_img/")
    Config.set("LABELING_DIRECTORY", "BACKGROUND_DIFF_DIR", "../data/frame/diff_background/")
    Config.set("LABELING_DIRECTORY", "LABEL_MAP_DIR", "../data/tf_data/")
    Config.set("LABELING_DIRECTORY", "ABNORMAL_ANNOTATED_FRAME_DIR", "../data/abnormal_annotated_frame/")

    Config.add_section("TFDATA_DIRECTORY")
    Config.set("TFDATA_DIRECTORY", "TFDATA_DIR", "../data/tf_data/")
    Config.set("TFDATA_DIRECTORY", "LABEL_MAP_DIR", "../data/tf_data/")
    Config.set("TFDATA_DIRECTORY", "MERGED_TFDATA_DIR", "../data/merged_tf/")
    Config.set("TFDATA_DIRECTORY", "SPLITTED_TFDATA_DIR", "../data/splitted_tf/")
    Config.set("TFDATA_DIRECTORY", "FRAME_DIR", "../data/frame/")
    Config.set("TFDATA_DIRECTORY", "FRAME_DIR_DIFF", "../data/frame/diff_background/")
    Config.set("TFDATA_DIRECTORY", "CSV_DIR", "../data/csv_data/")
    Config.set("TFDATA_DIRECTORY", "ANNOTATED_FRAME_DIR", "../data/annotated_frame/")
    Config.set("TFDATA_DIRECTORY", "TFRECORD_FILENAME", ".tfrecord")


    Config.add_section("LABELING_PARAMETER")
    Config.set("LABELING_PARAMETER", "MAX_AREA", "100")
    Config.set("LABELING_PARAMETER", "PADDING", "10")

    Config.add_section("SPLIT_PARAMETER")
    Config.set("SPLIT_PARAMETER", "TRAIN_SPLIT", "0.8")
    Config.set("SPLIT_PARAMETER", "TEST_SPLIT", "0.1")
    Config.set("SPLIT_PARAMETER", "VAL_SPLIT", "0.1")



    Config.add_section("DUMMY_DIRECTORY")
    Config.set("DUMMY_DIRECTORY", "DUMMY_TFDATA_PATH", "../data/tf_data/")

    Config.add_section("TRANSFORM_DIRECTORY")
    Config.set("TRANSFORM_DIRECTORY", "TRANSFORM_ANNOTATED_FRAME_DIR", "../data/annotated_frame/transform/")
    Config.set("TRANSFORM_DIRECTORY", "TRANSFORMED_FRAME_DIR", "../data/frame/transform/")

    Config.add_section("CLASSIFIER_DATA")
    Config.set("CLASSIFIER_DATA", "CROPPED_IMAGE_DIRECTORY", "../data/cropped_image_data/")
    Config.add_section("DATASET_NAME")
    Config.set("DATASET_NAME", "obj", "something")
    Config.write(cfgfile)
    cfgfile.close()
