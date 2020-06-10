import os
import gc
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
from keras import backend as K


from sklearn.preprocessing import LabelEncoder


# Import libraries for car detection
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())
# setup_logger()


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

    del cfg
    del outputs
    del predictor
    gc.collect()

    # Initialize list of car instances (list of dict)
    cars = extract_cars(carInstance, original)
    # WPOD-NET on car instances
    print('Loading WPOD-NET...')
    wpod_path = '../model/compressed-wpod.json'
    wpod_weights = '../model/wpod-net.h5'
    wpod_net = load_wpod(wpod_path, wpod_weights)
    cars = lp_recognition(cars, wpod_net)

    del wpod_net
    gc.collect()

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

    del model
    del labels
    gc.collect()

    # Mapping string back to original image
    print('Mapping results to input')
    new_image = lp2box(original, cars)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_name, new_image)


if __name__ == '__main__':
    main()
