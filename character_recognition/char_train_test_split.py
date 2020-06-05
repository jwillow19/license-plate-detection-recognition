import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from keras.applications import MobileNetV2
# Layers
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
# Compile
from keras.optimizers import Adam

# Load processed NumPy objects for train test split
print('Loading dataset...')
X = np.load('license_character_data.npy')
y = np.load('license_character_y.npy')

# Train/Test Split
print('Splitting data into train and test set...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1)

print('Train set shape: ', X_train.shape)
print('Train label shape: ', y_train.shape)
print('Test set shape: ', X_test.shape)
print('Test label shape: ', y_test.shape)

print('Saving...')
np.save('train/XTrain.npy', X_train)
np.save('train/yTrain.npy', y_train)
np.save('test/XTest.npy', X_test)
np.save('test/yTest.npy', y_test)
