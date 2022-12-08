import logging

import mpncov
import tensorflow as tf
from torchvision import transforms
import torch
from PIL import Image
import numpy as np


CLASSIFIER_FILENAME = '/meccano/real_data_classifier.pth.tar'
OBJECT_DETECTOR_PATH = '/meccano/real_data_od/saved_model'
DATASET_PATH = '/meccano/test_data.tfrecord'
DETECTOR_ONES_SIZE = (1, 480, 640, 3)
THRESHOLD = 0.4
LABELS = ['back', 'front', 'fronttwo', 'full', 'mid']

IMAGE_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
    ])

    classifier_representation = {
        'function': mpncov.MPNCOV,
        'iterNum': 5,
        'is_sqrt': True,
        'is_vec': True,
        'input_dim': 2048,
        'dimension_reduction': None,
    }

    freezed_layer = 0
    model = mpncov.Newmodel(classifier_representation,
                            len(LABELS), freezed_layer)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    trained_model = torch.load(CLASSIFIER_FILENAME)
    model.load_state_dict(trained_model['state_dict'])
    model.eval()

    detector = tf.saved_model.load(OBJECT_DETECTOR_PATH)
    ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
    detector(ones)

    good = 0
    bad = 0

    guesses_for_class = {
        outer_label: {inner_label: 0 for inner_label in LABELS}
        for outer_label in LABELS
    }

    for label in os.listdir(TEST_DIR):
        for images in os.listdir(os.path.join(TEST_DIR, label)):
            image_path = os.path.join(TEST_DIR, label, images)
            with open(image_path, 'rb') as f:
                tf_image = tf.image.decode_jpeg(f.read())

                detections = detector(np.expand_dims(tf_image.numpy(), 0))

                pil_img = Image.open(image_path)

                scores = detections['detection_scores'][0].numpy()
                boxes = detections['detection_boxes'][0].numpy()

                im_width, im_height = pil_img.size

                found_correct_label = False
                correct_label = label
                for score, box in zip(scores, boxes):
                    if score < THRESHOLD:
                        continue
                    logger.debug('found object')

                    ymin, xmin, ymax, xmax = box

                    (left, right, top, bottom) = (
                        xmin * im_width, xmax * im_width,
                        ymin * im_height, ymax * im_height)

                    cropped_pil = pil_img.crop((left, top, right, bottom))
                    transformed = transform(cropped_pil).cuda()

                    output = model(transformed[None, ...])
                    _, pred = output.topk(1, 1, True, True)
                    classId = pred.t()

                    label_name = LABELS[classId]
                    if label_name == correct_label:
                        found_correct_label = True

                    break

                guesses_for_class[correct_label][label_name] += 1
                if found_correct_label:
                    good += 1
                else:
                    print(label_name, correct_label)
                    bad += 1

    print('good:', good)
    print('bad:', bad)
    print(guesses_for_class)
    with open('guesses_for_class.pkl', 'wb') as f:
        pickle.dump(guesses_for_class, f)


if __name__ == '__main__':
    main()
