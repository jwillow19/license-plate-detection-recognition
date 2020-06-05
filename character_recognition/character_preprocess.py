import os
import cv2
import glob
import numpy as np

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical


# Get data
src_path = '../dataset_characters/'
char_data = glob.glob(os.path.join(src_path, '**/*.jpg'))
data_len = len(char_data)

# X - np.array of image
# labels - ground truth
X = []
labels = []
count = 1

print('Started...')
for data in char_data:
    print('Processing %i/%i' % (count, data_len))
    label = data.split('/')[-2]
    im = cv2.imread(data)
    im = cv2.resize(im, (80, 80))

    X.append(im)
    labels.append(label)
    count += 1

# Convert to numpy array
print('Converting to np.array...')
X = np.array(X, dtype='float16')
labels = np.array(labels)

print('Encoding Labels...')
# LabelEncoder
le = LabelEncoder()
# Fit labels and map each categorical to numerical value
le.fit(labels)
print('Label Classes: ', le.classes_)
# Transform [A,A,B,C,D,...,0,1,2,3] -> [0,0,1,2,3,...,36]
labels = le.transform(labels)
# One-Hot-Encode - Keras Utils library
print('One-Hot-Encoding...')
y = to_categorical(labels)

# save label file so we can use in another script
print('Saving label and training set to file...')
np.save('license_character_classes.npy', le.classes_)
np.save('license_character_data.npy', X)
np.save('license_character_y.npy', y)

# Train/Test Split
# print('Train/Test Split')
# X_train, y_train, X_test, y_test = train_test_split(
#     X, y, test_size=0.1, stratify=y, random_state=1)

print('done')
