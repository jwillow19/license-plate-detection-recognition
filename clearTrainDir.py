'''
Script to clear out contents in path
'''

import os

path = 'car_bbox_out/'
for filename in os.listdir(path):
    os.remove(path + filename)
