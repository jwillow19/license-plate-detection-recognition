'''
Reads results.csv and parses line-by-line, get bounding box for each line, then crop from the original image. Save cropped image as 'image_encoding_instance_#'

Time-Complexity: O(n) where n is the number of rows in results.csv, assuming cv2 operations are constant time

Space-Complexity: O(3) -> O(1) though I am not sure how much space pd.read_csv takes up
'''
import os
import cv2
import pandas as pd

# Create output directory if not exist
dst_dir = 'car_bbox_out/'
try:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
except OSError:
    print('Error: Creating bounding box output directory')

# Specify results.csv path and read DataFrame
output_path = 'pred_carBBox.csv'
results = pd.read_csv(output_path)

prev = ''
on = ''
instance_count = 1

# Iterate each row in DataFrame, crop and resize each instance
for ind, row in results.iterrows():
    try:
        # get row variables
        x, y, w, h = round(row['x']), round(
            row['y']), round(row['w']), round(row['h'])
        image_path, image_encoding = row['imgPath'], row['imgEncode']
        '''
        2-pointer method to iterate current and previous rows. Checks to see if
        current row is looking at instances in the same image as previous row.
        if yes, up instance_count, else reset instance_count to 1 
        '''
        prev = on
        on = image_encoding
        # base-case for first row
        if not prev:
            pass
        elif on == prev:
            instance_count += 1
        else:
            instance_count = 1

        im = cv2.imread(image_path)

        # crop and resize
        cropped_im = im[y:h, x:w]
        # cropped_im = cv2.resize(cropped_im, (224, 224))

        output_name = image_encoding + '_instance_' + \
            str(instance_count) + '.jpg'
        output_path = dst_dir + output_name
        print(output_path)

        cv2.imwrite(output_path, cropped_im)

    # Handle error thrown when x,y,w,h are NaN
    except ValueError:
        print('Skipping no detection row')
        pass

print('Finish cropping and resizing')
