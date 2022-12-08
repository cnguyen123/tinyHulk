import numpy as np
from matplotlib import pyplot as plt
import json
import cv2
import os
import Statistic_OD
import pandas as pd
import time
from textwrap import wrap
GROUNDTRUTH_JSON_FILE= "../data/statistic_data/tensorflow/auto/green/ground_true.json"
PREDICTION_JSON_FILE = "../data/statistic_data/tensorflow/auto/green/detection.json"

PHOTO_FOLDER = "../data/frame/green_clutt/"
photo_list =""
#photo_list = os.listdir(PHOTO_FOLDER)

with open(GROUNDTRUTH_JSON_FILE) as infile:
    gt_boxes = json.load(infile)
annotations = gt_boxes.get("annotations")

w1 = 640
h1 = 360
w2 = 1280
h2 = 720
def draw_box():
    for i in range(len(annotations)):
        id = int(annotations[i].get('id')) - 1
        image_id = annotations[i].get('image_id')

        file_name = "frame%05d.jpg" % id
        temp = annotations[i].get('bbox')
        # tensorflow coco return [xmin, ymin, w, h] --> transform to [xmin, ymin, xmax, ymax]
        bbox = [temp[0], temp[1], temp[0] + temp[2], temp[1] + temp[3]]

        img = cv2.imread(os.path.join(PHOTO_FOLDER, file_name))
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)
        new_file = os.path.join("../data/annotated_frame/green_clutt", photo_list[i])
        cv2.imwrite(new_file, img)


def draw_iou():
    with open(GROUNDTRUTH_JSON_FILE) as infile:
        gt_boxes = json.load(infile)
    annotations = gt_boxes.get("annotations")

    with open(PREDICTION_JSON_FILE) as infile:
        detect_boxes = json.load(infile)

    for i in range(len(detect_boxes)):
        item_i = detect_boxes[i]
        image_id = item_i.get('image_id')
        bbox = item_i.get('bbox')
        detect_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        id = -100
        gr_box = []
        for j in range(len(annotations)):
            anno_j = annotations[j]
            if anno_j.get('image_id') == image_id:
                id = int(anno_j.get('id')) - 1
                bbox = anno_j.get('bbox')
                gr_box = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
                break
        file_name = "frame%05d.jpg" % id
        file_path = os.path.join(PHOTO_FOLDER, file_name)
        img = cv2.imread(file_path)
        cv2.rectangle(img, tuple(gr_box[:2]),
                      tuple(gr_box[2:]), (0, 255, 0), 2)
        cv2.rectangle(img, tuple(detect_bbox[:2]),
                      tuple(detect_bbox[2:]), (0, 0, 255), 2)
        # compute the intersection over union and display it
        iou = Statistic_OD.intersection_over_union(gr_box, detect_bbox)
        cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        # save the output image
        cv2.imwrite(os.path.join("../data/iou_frame/green_clutt", file_name), img)

#return {image_id: {id:, pred_bboxes:[[b1],...bn], confidence_score[s1,...,sn], groundtrue_bbox:[], iou_scores:[i1, ...in]}}
def create_dict_of_detection_prediction():
    start_time = time.time()
    dict_return = {}

    with open("../data/statistic_data/tensorflow/auto/cluttered_green/detection.json") as infile:
        pd_boxes = json.load(infile)
    print('Total record in detection json ',len(pd_boxes))
    for i in range(len(pd_boxes)):
        image_id = pd_boxes[i].get('image_id')

        temp = pd_boxes[i].get('bbox')
        # coco return [xmin, ymin, w, h] --> transform to [xmin, ymin, xmax, ymax]
        bbox = [temp[0], temp[1], temp[0] + temp[2], temp[1] + temp[3]]

        score = pd_boxes[i].get('score')
        if image_id in dict_return.keys():
            # print("Exist key!")
            dict_return.get(image_id).get('pred_bboxes').append(bbox)
            dict_return.get(image_id).get('confidence_scores').append(score)
        else:
            bbox_list = [bbox]
            scores_list = [score]
            dict_return[image_id] = {'pred_bboxes': bbox_list, 'confidence_scores': scores_list}

    #print(dict_return.get(list(dict_return.keys())[0]))
    with open("../data/statistic_data/tensorflow/auto/cluttered_green/ground_true.json") as infile:
        gt_boxes = json.load(infile)
    annotations = gt_boxes.get("annotations")
    total_image = len(annotations)
    print('total record in ground_true ', total_image)
    for j in range(len(annotations)):
        update_dict = {}
        anno_j = annotations[j]
        image_id = anno_j.get('image_id')
        id = int(anno_j.get('id')) - 1
        bbox = anno_j.get('bbox')
        gr_box = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        temp = dict_return.get(image_id)
        temp['id'] = id
        temp['groundtrue_bbox'] = gr_box
        iou_list = []
        detect_bboxes = temp.get('pred_bboxes')
        for k in range(len(detect_bboxes)):
            detect_box_k = detect_bboxes[k]
            iou_k = Statistic_OD.intersection_over_union(gr_box, detect_box_k)
            iou_list.append(iou_k)
        temp['iou_scores'] = iou_list
        update_dict[image_id] = temp
        dict_return.update(update_dict)

    #print(dict_return.get(list(dict_return.keys())[0]))
    df = pd.DataFrame(columns=['Images', 'Score', 'TP', 'FP', 'ACC_TP', 'ACC_FP', 'Precision', 'Recall'])
    IOU_thresh = 0.5
    for image_id in dict_return.keys():
        item_i = dict_return.get(image_id)
        confidence_scores = item_i.get('confidence_scores')
        iou_scores = item_i.get('iou_scores')
        id = item_i.get('id')
        for m in range(len(iou_scores)):
            iou = iou_scores[m]
            if iou >= IOU_thresh:
                TP = 1
                FP = 0
                df = df.append(
                    {'Images': str(id), 'Score': confidence_scores[m], 'TP': TP, 'FP': FP, 'ACC_TP': 0, 'ACC_FP': 0, 'Precision': 0,
                     'Recall': 0},
                    ignore_index=True)
            else:
                FP = 1
                TP = 0
                df = df.append(
                    {'Images': str(id), 'Score': confidence_scores[m], 'TP': TP, 'FP': FP, 'ACC_TP': 0, 'ACC_FP': 0, 'Precision': 0,
                     'Recall': 0},
                    ignore_index=True)
    print("Finish calculating TP FP, now calculating ACC_TP, ACC_FP...")
    #print(df)
    df = df.sort_values('Score', ascending=False).reset_index()

    # update ACC_TP, ACC_FP
    ACC_TP = 0
    ACC_FP = 0
    for i in range(df.shape[0]):
        TP = int(df.at[i, 'TP'])
        FP = int(df.at[i, 'FP'])
        ACC_FP = ACC_FP + FP
        ACC_TP = ACC_TP + TP
        df.loc[i, ['ACC_TP', 'ACC_FP']] = [ACC_TP, ACC_FP]

        precision = ACC_TP / (ACC_TP + ACC_FP)
        recall = ACC_TP / total_image
        if recall > 1:
            recall = 1
        df.loc[i, ['Precision', 'Recall']] = [precision, recall]

    csv_pr_data_filename = "../data/statistic_data/pr_curve_csv/" + "auto" + "/" + 'cluttered_green' + '/temp.csv'
    df.to_csv(csv_pr_data_filename)
    print("Finish calculating ACC_TP, ACC_FP... Now making interpolated dataset...")


    Precisions = df['Precision']
    Recalls = df['Recall']
    data_interpolated = pd.DataFrame(columns=['In_Precisions', 'In_Recalls'])

    for i in range(len(Recalls)):
        data_interpolated = data_interpolated.append(
            {'In_Precisions': max(Precisions[i:len(Precisions)]), 'In_Recalls': Recalls[i]}, ignore_index=True)

    # print(data_interpolated)
    csv_interpolated_pr_data_filename = "../data/statistic_data/pr_curve_csv/" + 'auto' + "/" + 'cluttered_green' + '/interpolated_temp.csv'
    data_interpolated.to_csv(csv_interpolated_pr_data_filename)
    print("Save statistic data to csv file succsefully. Now call the draw_pr_curve to draw PR curve")

    end_time = time.time()
    print('Running time {:.4f} secs'.format(end_time - start_time))
    return dict_return








def generating_PR_dataset(iou_thresh=0.5, test_set_name = "cluttered_green", model_name = "auto"):
    directory_name = "../data/statistic_data/pr_curve_csv/" + model_name + "/" + test_set_name
    print(directory_name)
    if not os.path.exists(directory_name):
        os.umask(0)

        os.makedirs(directory_name, mode=0o777)
    start_time = time.time()
    print("Generating PR data for model {} on test set {}...".format(model_name, test_set_name))
    GROUNDTRUTH_JSON = "../data/statistic_data/tensorflow/" + model_name+ "/" + test_set_name + "/ground_true.json"
    PREDICTION_JSON = "../data/statistic_data/tensorflow/" + model_name + "/" + test_set_name + "/detection.json"
    with open(GROUNDTRUTH_JSON) as infile:
        gt_boxes = json.load(infile)
    annotations = gt_boxes.get("annotations")
    Precisions = []
    Recalls = []
    with open(PREDICTION_JSON) as infile:
        detect_boxes = json.load(infile)

    df = pd.DataFrame(columns=['Images', 'Score', 'TP', 'FP', 'ACC_TP', 'ACC_FP', 'Precision', 'Recall'])

    print("total record in detection:", len(detect_boxes))
    print("total records in groundtrue:", len(annotations))
    total_image = len(annotations)

    #dict keeps all image_id has been retrieved so that no need to loop through the original ground_true annotations
    annotations_retrieved = {} #{image_id: {id:, bbox:[]}}
    for i in range(len(detect_boxes)):
        item_i = detect_boxes[i]
        image_id = item_i.get('image_id')
        bbox = item_i.get('bbox')
        detect_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        score = float(item_i.get('score'))
        id = -100
        gr_box = []

        if image_id in annotations_retrieved.keys():
            id = annotations_retrieved.get(image_id)[0]
            gr_box = annotations_retrieved.get(image_id)[1]
            #gr_box = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        else:
            for j in range(len(annotations)):
                anno_j = annotations[j]
                if anno_j.get('image_id') == image_id:
                    id = int(anno_j.get('id')) - 1
                    bbox = anno_j.get('bbox')
                    gr_box = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
                    annotations_retrieved[image_id] = [id, gr_box]
                    annotations.remove(anno_j)
                    break


        # for j in range(len(annotations)):
        #     anno_j = annotations[j]
        #     if anno_j.get('image_id') == image_id:
        #         id = int(anno_j.get('id')) - 1
        #         bbox = anno_j.get('bbox')
        #         gr_box = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        #         break

        # compute the intersection over union
        iou = Statistic_OD.intersection_over_union(gr_box, detect_bbox)
        if iou >= iou_thresh:
            TP = 1
            FP = 0
            df = df.append({'Images': str(id), 'Score': score, 'TP': TP, 'FP': FP, 'ACC_TP':0, 'ACC_FP':0, 'Precision':0, 'Recall':0},
                           ignore_index=True)
        else:
            FP = 1
            TP = 0
            df = df.append(
                {'Images': str(id), 'Score': score, 'TP': TP, 'FP': FP, 'ACC_TP': 0, 'ACC_FP': 0, 'Precision': 0, 'Recall': 0},
                ignore_index=True)

    print("Finish calculating TP FP, now calculating ACC_TP, ACC_FP...")
    df = df.sort_values('Score', ascending=False).reset_index()

    #update ACC_TP, ACC_FP
    ACC_TP = 0
    ACC_FP = 0
    for i in range(df.shape[0]):
        TP = int(df.at[i, 'TP'])
        FP = int(df.at[i, 'FP'])
        ACC_FP = ACC_FP + FP
        ACC_TP = ACC_TP + TP
        df.loc[i, ['ACC_TP', 'ACC_FP']] = [ACC_TP, ACC_FP]

        precision = ACC_TP/(ACC_TP + ACC_FP)
        recall = ACC_TP/total_image
        if recall > 1:
            recall = 1
        df.loc[i, ['Precision', 'Recall']] =[precision, recall]

    print("Finish calculating ACC_TP, ACC_FP... Now making interpolated dataset...")


    csv_pr_data_filepath = os.path.join(directory_name, 'statistic_IOU_{}.csv'.format(iou_thresh))
    df.to_csv(csv_pr_data_filepath)
    Precisions = df['Precision']
    Recalls = df['Recall']

    #plt.plot(Recalls, Precisions, linewidth=2, color="red")
    #plt.xlabel("Recall", fontsize=12, fontweight='bold')
    #plt.ylabel("Precision", fontsize=12, fontweight='bold')
    #plt.title("Precision-Recall Curve. IOU_thresh = {}".format(iou_thresh), fontsize=15, fontweight="bold")
    #plt.show()

    #draw interpolated PR-curve
    data_interpolated = pd.DataFrame(columns=['In_Precisions', 'In_Recalls'])

    for i in range(len(Recalls)):
        data_interpolated = data_interpolated.append( {'In_Precisions': max(Precisions[i:len(Precisions)]), 'In_Recalls': Recalls[i]},ignore_index=True)

    #print(data_interpolated)

    csv_interpolated_pr_data_filepath = os.path.join(directory_name, 'interpolated_statistic_IOU_{}.csv'.format(
        iou_thresh))

    data_interpolated.to_csv(csv_interpolated_pr_data_filepath)
    print("Save statistic data to csv file succsefully. Now call the draw_pr_curve to draw PR curve")

    end_time = time.time()
    print('Running time {:.4f} secs'.format(end_time - start_time))



def draw_pr_curve(iou_thresh = 0.75, model_name = "auto", test_set_name = "cluttered_green"):
    pr_data_filename = "../data/statistic_data/pr_curve_csv/" + model_name + "/" + test_set_name + '/statistic_IOU_{}.csv'.format(iou_thresh)
    interpolated_pr_data_filename = "../data/statistic_data/pr_curve_csv/" + model_name + "/" + test_set_name + '/interpolated_statistic_IOU_{}.csv'.format(
        iou_thresh)

    #pr_data_filename = "../data/statistic_data/pr_curve_csv/" + model_name + "/" + test_set_name + '/temp.csv'
    #interpolated_pr_data_filename = "../data/statistic_data/pr_curve_csv/" + model_name + "/" + test_set_name + '/interpolated_temp.csv'
    data_main = pd.read_csv(pr_data_filename)
    interpolated_data = pd.read_csv(interpolated_pr_data_filename)
    Precisions = data_main['Precision']
    Recalls = data_main['Recall']
    in_precisions = interpolated_data['In_Precisions']
    in_recalls = interpolated_data['In_Recalls']

    plt.plot(Recalls, Precisions, linewidth=2, color="red", label="Precision")
    plt.plot( in_recalls, in_precisions, linewidth=2, color="blue",
             label="Interpolated Precision", linestyle='dashed')
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("\n".join(wrap(
        "Precision-Recall Curve. \nModel: {}. IOU_thresh = {}. Test_set = {}".format(model_name, iou_thresh, test_set_name), 60)))

    #plt.title("Precision-Recall Curve. IOU_thresh = {}. Data = {}".format(iou_thresh, test_set_name), fontsize=12, fontweight="bold")
    plt.tight_layout()
    #plt.subplots_adjust(top=0.8)
    plt.legend()
    plt.show()

#for a in ["cluttered_green", "cluttered_table", "table", "green"]:
#    print(a)
#    generating_PR_dataset(iou_thresh=0.75, model_name="auto", test_set_name= a)
generating_PR_dataset(iou_thresh=0.9, model_name="auto", test_set_name="meccano_bike_green_diff")
#draw_pr_curve(iou_thresh=0.75, model_name="auto", test_set_name="green")
#create_dict_of_detection_prediction()
#generating_PR_dataset()

draw_pr_curve(iou_thresh=0.9, model_name="auto", test_set_name="meccano_bike_green_diff")



