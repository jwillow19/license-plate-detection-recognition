import os
import glob
import pandas as pd
import shutil

# Goal - same to csv file
#   each image instance - path to image, bbox coords
# 1. Get all .jpg format images

class_type = 'Car'
src_dir = "train/" + class_type
dst_dir = "test/" + class_type

try:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
except OSError:
    print('Error: Creating test directory')

# Iterate through all .jpg files, make copy of jpg and annotated txt file to test dir
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    print(jpgfile)
    annotate_file = jpgfile.split('.')[0] + '.txt'
    print(annotate_file)
    shutil.copy(jpgfile, dst_dir)
    shutil.copy(annotate_file, dst_dir)

    # Open annotated_text file and save each line to DF
    txt_file = open()
