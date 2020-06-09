# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long
import os
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


from keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder


# Import libraries for car detection
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())
# setup_logger()

# import some common detectron2 utilities


class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1-tl1, br2-tl2
    assert((wh1 >= 0).all() and (wh2 >= 0).all())

    intersection_wh = np.maximum(np.minimum(
        br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area/union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

# Reconstruction function from predict value into plate cropped from image


def reconstruct(I, Iresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2
    net_stride = 2**4
    side = ((208 + 40)/2)/net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)
    # CNN input image size
    WH = getWH(Iresized.shape)
    # output feature map size
    MN = WH/net_stride

    vxx = vyy = 0.5  # alpha
    def base(vx, vy): return np.matrix(
        [[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        pts_frontal = np.array(B*base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    # print(final_labels_frontal)
    try:
        # LP size and type
        out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[
            0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

        TLp = []
        Cor = []
        if len(final_labels):
            final_labels.sort(key=lambda x: x.prob(), reverse=True)
            for _, label in enumerate(final_labels):
                t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
                ptsh = np.concatenate(
                    (label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
                H = find_T_matrix(ptsh, t_ptsh)
                Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
                TLp.append(Ilp)
                Cor.append(ptsh)
        return final_labels, TLp, lp_type, Cor

    except IndexError as e:
        print('(reconstruct) No LP detected: ', e)
        return


def detect_lp(model, I, max_dim, lp_threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img
    w, h = (np.array(I.shape[1::-1], dtype=float)
            * factor).astype(int).tolist()
    Iresized = cv2.resize(I, (w, h))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)
    try:
        L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
        return L, TLp, lp_type, Cor
    except TypeError as e:
        print('(detect_lp) Caught TypeError: ', e)
        pass


# Pipeline functions
'''
Car Detection Module
'''


def cfg_detectron():
    '''
    Detectron2 configuration
    Output: Returns a predictor using FasterRCNN
    '''
    cfg = get_cfg()
    # Go to model_zoo and choose a config file for a model - Go to GitHub
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def prune_class(outputs):
    '''
    Function prunes classes from predicted output to only contain class 'car'
    Input: predicted output
    Output: NewInstance with only class, scores, bboxes from car instances
    '''
    # 1. Make variable reference to output
    img_size = outputs["instances"].image_size
    cls = outputs["instances"].pred_classes
    scores = outputs['instances'].scores
    boxes = outputs['instances'].pred_boxes

    # 2. Remove non-car classes - car class = 2?
    # flatten tensor to list
    cls_to_list = cls.flatten().tolist()
    # list comprehension to build list with index wherever value != 2 in class list
    indx_to_remove = [ind if val != 2 else None for (
        ind, val) in enumerate(cls_to_list)]
    # filter None and return list
    indx_to_remove = list(filter(lambda x: isinstance(x, int), indx_to_remove))

    # 3. Delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    scores = np.delete(scores.cpu().numpy(), indx_to_remove)
    # WORKAROUND for numpy.ndarray - loops through boxes.tensor and remove tensor
    # index-wise, stack bbox tensors back into one tensor
    car_boxes = []
    for ind, tensor in enumerate(boxes.tensor):
        if ind not in indx_to_remove:
            car_boxes.append(tensor)
        else:
            continue

    # Initialize cars_list: a list of dictionary to store all cars instances
    cars_list = []
    try:
        car_box_tensor = torch.stack(car_boxes)
        bbox_list = car_box_tensor.tolist()

        # for-loop to save imgsize, cls, scores, car bbox instances
        for ind, instance in enumerate(cls):
            bbox = bbox_list[ind]
            x, y, w, h = bbox

            car = {
                #             'imgPath': jpgfile,
                #             'imgEncode': imgEncoding,
                'class': 'car',
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            }

            cars_list.append(car)

        # 4. Convert back to tensor and move to cuda
        cls = torch.tensor(cls).to('cuda:0')
        scores = torch.tensor(scores).to('cuda:0')
        boxes.tensor = car_box_tensor

        # 5. Create new instance object and set its fields
        obj = detectron2.structures.Instances(image_size=img_size)
        obj.set('pred_classes', cls)
        obj.set('scores', scores)
        obj.set('pred_boxes', boxes)

        newInstance = {'instances': obj}

        return newInstance, cars_list

   # RuntimeError when torch.stack fail to stack empty car_box_tensor array
    except RuntimeError:

        return {}, cars_list


'''
WPOD-NET MODULE
# preprocess_input (Preprocess car image input)
# detect_plate (detects LP given an image)
# load_model (loads pretrained wpod-net)
'''


def load_model(model_path, model_weights):
    '''
    Input (STR): path to model.json, model.h5
    reads the model path, get model architecture from .json then load
    the weights from .h5 
    '''
    try:
        # parse json string to initialize model instance
        with open(model_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects=None)
        # load model weights
        model.load_weights(model_weights)
        print('Loaded WPOD-NET successfully...')

        return model
    except Exception as e:
        print(e)


def preprocess_input(image, resize=False):
    '''
    Function preprocess the image input before feeding it to WPOD-NET
    Input: An image (np.ndarray), (Bool) resize
    Output: processed version of image
    '''
    # im = cv2.imread(img_path)
    # convert to RGB
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # standardize pixels between 0 and 1
    im = im / 255
    # resize: if dim not (224,224)
    if resize:
        im = cv2.resize(im, (224, 224))
    return im


def detect_plate(image, model):
    '''
    Input: An image (np.ndarray)
    Function calculates the Resizing-Factor mentioned in the paper
    Dmin = 288 and Dmax = 608 are chosen such that it produces optimal
        results between accuracy and running time
    Output: Lp image and coordinates for detected LP
    '''
    Dmax = 608
    Dmin = 288
    bbox = preprocess_input(image)
    # get width:height image ratio
    ratio = float(max(bbox.shape[:2]) / min(bbox.shape[:2]))
    size = int(ratio*Dmin)
    bound = min(size, Dmax)
    # Detect LP with wpod-net
    try:
        _, LpImg, _, cor = detect_lp(model, bbox, bound, lp_threshold=0.5)
        return LpImg, cor
    except TypeError:
        print('No LP detected, return None')
        return None, None


def lp_recognition(cars, model):
    '''
    Input: List of dictionarys (key:id, val:np.ndarray images)
    Output_new: modify the cars array with additional LP field in dictionary
    Output: List of dictionarys np.ndarray images of detected LPs
    '''
    for car in cars:
        try:
            im = car['image']
            LpImg, _ = detect_plate(im, model)
            car['lp'] = LpImg[0]

        except TypeError:
            car['lp'] = None

    return cars


'''
OCR Module
# Modified MobileNetv2 Module: Preprocess -> get contours -> sort contours -> cropped characters
# process_lp (Preprocess LP for segmentation)
# sort_contours (Sort a list of contours)
'''


def process_lp(image):
    '''
    Function blurs, de-noise, binary, dilate image
    Input: np.ndarray
    '''
    # Convert from (0,1) to (1,255)
    im = cv2.convertScaleAbs(image, alpha=(255.0))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (5, 5), 0)
#     _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    im = cv2.adaptiveThreshold(
        im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    dilation_filter3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    im = cv2.dilate(im, dilation_filter3, iterations=1)

    return im


def sort_contours(cnts, reverse=False):
    '''
    Function sorts a list of contours for each segmented character
    Input (list of contours (list)) and grab each contour bbox 
    '''
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def segmentation(image, ratio_lower=1, ratio_upper=5):
    '''
    Function enbodies all procedures in character segmentation
    Input: An image (np.ndarray)
    Output: a list of cropped image characters 
    '''
    # Perform image processing on input LP
    processed_im = process_lp(image)
    contour, _ = cv2.findContours(
        processed_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize a list which will be used to append charater image
    crop_characters = []
    digit_w, digit_h = 30, 60

    # Loop through the list of contours to get bounding box on character
    for c in sort_contours(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w

#         print(ratio)

        # Select contours that satisfy both conditions
        if ratio_lower <= ratio <= ratio_upper and h/processed_im.shape[0] >= 0.2:

            # Seperate number and give prediction
            curr_num = processed_im[y:y+h, x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY)
            crop_characters.append(curr_num)

    return crop_characters

# Load Recognition model architecture, weights and classes


def load_recognition(json_model, model_weights, labels_file):
    '''
    Function: Loads recognition model with architecture and weights, and load class labels
    Input: path to model architecture, weights and class labels files
    '''
    with open(json_model, 'r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights(model_weights)

    labels = LabelEncoder()
    labels.classes_ = np.load(labels_file)

    return model, labels


def predict_lp_chars(crop_characters, model, labels):
    '''
    Function: Takes a list of contours and run recognition on each character, return a concatenated string 
        of each detected character
    Input: list of contours from segmentation module
    Output: string representing the detected contour characters
    '''
    lp_chars = ''

    for i, char in enumerate(crop_characters):

        # Preprocess image to feed to model: resize, stack dimensions, add newaxis
        # Model expected input: (m, w, h. c)
        im = cv2.resize(char, (128, 128))
        im = np.stack((im,)*3, axis=-1)
        im = im[np.newaxis, :]

        result = labels.inverse_transform([np.argmax(model.predict(im))])

        detected_char = np.array2string(result)
        detected_char = detected_char.strip("'[]")
        lp_chars += detected_char

    return lp_chars
