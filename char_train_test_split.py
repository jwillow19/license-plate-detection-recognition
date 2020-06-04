import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Load processed NumPy objects for train test split
X = np.load('license_character_data.npy')
y = np.load('license_character_y.npy')

# Train Test split
# Train/Test Split
X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1)
