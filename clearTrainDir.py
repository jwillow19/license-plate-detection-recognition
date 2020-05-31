'''
Script to clear out contents in path
'''

import os

path = './train/Car/'
for filename in os.listdir(path):
    os.remove(path + filename)
