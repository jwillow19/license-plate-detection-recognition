'''
Character Segmentation Module
Get the WPOD-NET outputs from 'lp_out/' directory and process these images before sending it for OCR
'''
import os
from os.path import splitext, basename
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import detect_lp

import tensorflow as tf
import keras
from keras.models import model_from_json

src_path = 'lp_out/'

for lp_file in glob.iglob(os.path.join(src_path, "*.jpg")):
    print(lp_file)
