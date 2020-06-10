import torchvision
import torch
# from sklearn.preprocessing import LabelEncoder
# from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
import streamlit as st
import os
import json
import gc
# import argparse
# import detectron2
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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Move functions into utils
from utils import *


# Import libraries for car detection
# print(torch.__version__, torch.cuda.is_available())


@st.cache
def load_wpod(json_model, model_weights):
    '''
    Input (STR): path to model.json, model.h5
    reads the model path, get model architecture from .json then load
    the weights from .h5 
    '''
    try:
        # parse json string to initialize model instance
        with open(json_model, 'r') as f:
            # model_json = json_file.read()
            model = model_from_json(f.read())

        # load model architecture and weights
        model.load_weights(model_weights)
        print('Loaded WPOD-NET successfully...')

        return model
    except Exception as e:
        st.write(e)


@st.cache
def load_detectron():
    cfg, predictor = cfg_detectron()
    return cfg, predictor


def main():
    st.title('License Plate Recognizer')
    wpod_net = load_wpod(
        '../model/compressed-wpod.json', '../model/wpod-net.h5')
    cfg, predictor = load_detectron()

    st.write('To try out this project, upload a image')
    uploaded_file = st.file_uploader(
        "Choose a jpg file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        st.image(
            image,
            caption="Image has shape: {}".format(image_array.shape),
            use_column_width=True
        )

        # Detectron2
        outputs = predictor(image_array)
        carInstance, _ = prune_class(outputs)
        del cfg
        del outputs
        del predictor
        gc.collect()

        # v = Visualizer(image_array[:, :, ::-1],
        #                MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # v = v.draw_instance_predictions(carInstance["instances"].to("cpu"))

        # st.image(
        #     v.get_image()[:, :, ::-1],
        #     caption="Vehicle detected in image",
        #     use_column_width=True
        # )

        # wpod_net = load_wpod(
        #     '../model/compressed-wpod.json', '../model/wpod-net.h5')
        cars = extract_cars(carInstance, image_array)
        cars = lp_recognition(cars, wpod_net)

        del wpod_net
        gc.collect()

        path_to_arch = '../character_recognition/mobile_base.json'
        path_to_weights = '../character_recognition/License_character_recognition.h5'
        path_to_labels = '../character_recognition/license_character_classes.npy'
        model, labels = load_recognition(
            path_to_arch, path_to_weights, path_to_labels)
        cars = get_chars(cars, model, labels)
        del model
        del labels
        gc.collect()

        new_image = lp2box(original, cars)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

        st.image(
            new_image,
            caption="Final",
            use_column_width=True
        )


if __name__ == '__main__':
    main()
