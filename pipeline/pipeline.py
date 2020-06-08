'''
Process the LP and returns a list of contours for each character
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


def process_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (7, 7), 0)
    _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)

    dilation_filter3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    im = cv2.dilate(im, dilation_filter3, iterations=1)

    return im


def sort_contours(cnts, reverse=False):
    '''
    Input (list of contours (list)) and grab each contour bbox 
    '''
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


digit_w, digit_h = 30, 60

# I DONT THINK I SHOULD SAVE CONTOUR - INSTEAD GET CONTOUR IN REAL TIME GIVEN IMAGE
for jpgfile in glob.iglob(os.path.join('../lp_out/', '*.jpg')):

    crop_characters = []

    path = jpgfile
    encoding = jpgfile.split('.')[2].split('/')[-1]

    im = process_image(jpgfile)
    contour, _ = cv2.findContours(
        im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
