import os
import argparse
import detectron2
# from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json
from os.path import splitext, basename
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Move functions into utils
from utils import *

from keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder


# Import libraries for car detection
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())
# setup_logger()


def extract_cars(output_instance, image):
    '''
    Input: Original image (Input for FasterRCNN) and predicted output class Instance
    Output: A list of dictionary of (key:id, val:cropped images of detected car)
    '''
    bboxes = output_instance['instances'].pred_boxes

    cars = []

    for ind, instance in enumerate(bboxes.tensor.tolist()):
        # pretty elegant eh? Round each coordinate to int and destructure
        x, y, w, h = [round(num) for num in instance]
        # Crop car instance and save to array
        cropped_im = image[y:h, x:w]
        car = {
            'id': ind,
            'image': cropped_im,
            'car_bbox': [x, y, w, h]
        }
        cars.append(car)

    return cars


def prune_empty(cars):
    '''
    Input: list of dictionarys
    Output: modified input - list of dictionarys with no empty LP value
    '''
    for ind, car in enumerate(cars):
        if car['lp'] is None:
            cars.pop(ind)
    return cars


def get_chars(cars, model, labels):
    '''
    Function takes in a list of LPs and return a list of dictionary (key: id, val: nparray of chars)
    Input: List of dictionary (of car instance), model for recognition, labels: class labels 
    Output: Modify input - List of dictionary with chars
    '''
    for car in cars:
        crop_characters = segmentation(car['lp'])
        lp_string = predict_lp_chars(crop_characters, model, labels)

        car['chars'] = lp_string

    return cars


def lp2box(image, cars):
    '''
    Input: An image original, np.ndarray; cars (list of dictionary with car instance information(id, lp, car_bbox, image, chars))
    Output: An image, with bbox around cars and LP character text
    '''

    new_image = np.copy(image)

    for car in cars:
        lp_text = car['chars']
        x, y, w, h = car['car_bbox']

        # Initialize random color
        color = (int(np.random.randint(100, 255, 1, int)), int(
            np.random.randint(100, 255, 1, int)), int(np.random.randint(100, 255, 1, int)))
        # Draw bbox to original image
        new_image = cv2.rectangle(
            new_image, (x, y), (w, h), color, thickness=2)
        # Put string text on image
        cv2.putText(new_image, lp_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=color, thickness=2)

    return new_image


'''
Pipeline
1. Get a scene
2. Run FasterRCNN on scene -> Save Coordinates of bbox instances -> Crop -> Save to array
3. Pass array onto WPOD-NET, run model on each cropped instance
4. Run OCR on each plate -> Save LP onto a list or hastable
5. Append LP back to car bbox
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../car_detection/test/Car/be41c99204906f64.jpg',
                        help='Path to input image')
    parser.add_argument('--out', type=str, default='final.jpg',
                        help='Name of output image')

    args = parser.parse_args()

    input_path, out_name = args.input, args.out
    original = cv2.imread(input_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # FasterRCNN - configure and predict
    print('Configuring Detectron2...')
    cfg, predictor = cfg_detectron()
    outputs = predictor(original)
    carInstance, _ = prune_class(outputs)

    # Initialize list of car instances (list of dict)
    cars = extract_cars(carInstance, original)
    # WPOD-NET on car instances
    print('Loading WPOD-NET...')
    wpod_net = load_model('../model/wpod-net.json', '../model/wpod-net.h5')
    cars = lp_recognition(cars, wpod_net)
    # Remove car instances with empty LP from list of car instances
    cars = prune_empty(cars)

    # Segmentation and Recogition
    print('Loading Recognition Module...')
    path_to_arch = '../character_recognition/mobile_base.json'
    path_to_weights = '../character_recognition/License_character_recognition.h5'
    path_to_labels = '../character_recognition/license_character_classes.npy'
    model, labels = load_recognition(
        path_to_arch, path_to_weights, path_to_labels)
    cars = get_chars(cars, model, labels)

    # Mapping string back to original image
    print('Mapping results to input')
    new_image = lp2box(original, cars)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_name, new_image)


if __name__ == '__main__':
    main()
