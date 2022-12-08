import cv2
from matplotlib import pyplot as plt
import config
import numpy as np
import sklearn.metrics
#from pycocotools.cocoeval import COCOeval
#from pycocotools.coco import COCO
import os
import json
GROUNDTRUTH_JSON_FILE = "../data/statistic_data/ground_true.json"
PHOTO_FOLDER = "../data/frame/green_clutt/"
PREDICTION_JSON_FILE = "../data/statistic_data/detection.json"
def intersection_over_union(gt, pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])
    # if there is no overlap between predicted and ground-truth box
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def plt_imshow(title, image, path):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.axis("off")
    plt.imsave(path, image)
    plt.show()

def load_data_from_json():

    photo_list = os.listdir(PHOTO_FOLDER)

    with open(GROUNDTRUTH_JSON_FILE) as infile:
        gt_boxes = json.load(infile)
    annotations = gt_boxes.get("annotations")

    with open(PREDICTION_JSON_FILE) as infile:
        dectect_boxes = json.load(infile)


    for i in range(len(annotations)):
        id = int(annotations[i].get('id')) - 1
        image_id = annotations[i].get('image_id')

        file_name = "frame%05d.jpg" % id
        temp = annotations[i].get('bbox')
        # tensorflow coco return [xmin, ymin, w, h] --> transform to [xmin, ymin, xmax, ymax]
        bbox = [temp[0], temp[1], temp[0] + temp[2], temp[1] + temp[3]]
def compute_iou(imagePath):
    # load the image
    image = cv2.imread(imagePath)
    # define the top-left and bottom-right coordinates of ground-truth
    # and prediction


    #xmin,ymin, xmax, ymax
    groundTruth = [156, 101, 156+ 783, 101 + 297]
    prediction = [158, 101, 158 + 784, 101 + 299]
    # draw the ground-truth bounding box along with the predicted
    # bounding box
    cv2.rectangle(image, tuple(groundTruth[:2]),
        tuple(groundTruth[2:]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(prediction[:2]),
        tuple(prediction[2:]), (0, 0, 255), 2)
    # compute the intersection over union and display it
    iou = intersection_over_union(groundTruth, prediction)
    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    # show the output image
    plt_imshow("Image", image, config.IOU_RESULT)


def compute_precision_recall(yTrue, predScores, thresholds):
    precisions = []
    recalls = []
    # loop over each threshold from 0.2 to 0.65
    for threshold in thresholds:
        # yPred is dog if prediction score greater than threshold
        # else cat if prediction score less than threshold
        yPred = [
            "dog" if score >= threshold else "cat"
            for score in predScores
        ]

        # compute precision and recall for each threshold
        precision = sklearn.metrics.precision_score(y_true=yTrue,
                                                    y_pred=yPred, pos_label="dog")
        recall = sklearn.metrics.recall_score(y_true=yTrue,
                                              y_pred=yPred, pos_label="dog")

        # append precision and recall for each threshold to
        # precisions and recalls list
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))
    # return them to calling function
    return precisions, recalls


def pr_compute():
    # define thresholds from 0.2 to 0.65 with step size of 0.05
    thresholds = np.arange(start=0.2, stop=0.7, step=0.05)
    # call the compute_precision_recall function
    precisions, recalls = compute_precision_recall(
        yTrue=config.GROUND_TRUTH_PR, predScores=config.PREDICTION_PR,
        thresholds=thresholds,
    )

    # return the precisions and recalls
    return (precisions, recalls)

def plot_pr_curve(precisions, recalls, path):
    # plots the precision recall values for each threshold
    # and save the graph to disk
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig(path)
    plt.show()

#combine precision and recall into a single metric by taking their harmonic mean.
# A higher F1-score would mean that precision and recall are high,
# while a lower F1-score signifies a high imbalance between precision and recall
def f1_score(precisions, recalls):
    f1_score = np.divide(2 * (np.array(precisions) * np.array(recalls)), (np.array(precisions) + np.array(recalls)))
    return f1_score

#cocoAnnotation = COCO(annotation_file="../data/statistic_data/ground_true.json")
#cocovalPrediction = cocoAnnotation.loadRes("../data/statistic_data/detection.json")
# initialize the COCOeval object by passing the coco object with
# ground truth annotations, coco object with detection results
#cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")

# run evaluation for each image, accumulates per image results
# display the summary metrics of the evaluation
#cocoEval.evaluate()
#cocoEval.accumulate()
#cocoEval.summarize()