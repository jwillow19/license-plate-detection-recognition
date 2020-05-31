'''
Reads test/<class> dir and parse each annotated text file line by line, then save each line as an instance in a DataFrame 
'''
import os
import glob
import pandas as pd
import shutil

path = 'test/Car'
rows_list = []

for txtFile in glob.iglob(os.path.join(path, "*.txt")):
    # print(txtFile)

    image_path = txtFile.split('.')[0] + '.jpg'
    image_enc = image_path.split('.')[0].split('/')[-1]

    # open .txt and readlines - outputs array with lines
    txt_file = open(txtFile, 'r')
    lines = txt_file.readlines()

    for line in lines:
        '''
        line format - 'class,x,y,w,h' (x,y) top left coords, and (w,h) bottom right coordinates
        '''
        # gt_table.loc[index] =
        # line.rstrip('\n')
        line_class = line.split(',')[0]
        line_x = line.split(',')[1]
        line_y = line.split(',')[2]
        line_w = line.split(',')[3]
        line_h = line.split(',')[4].rsplit('\n')[0]

        # initialize hashtable to store row information
        row_dict = {
            'imgPath': image_path,
            'imgEncode': image_enc,
            'class': line_class,
            'x': line_x,
            'y': line_y,
            'w': line_w,
            'h': line_h,
        }

        rows_list.append(row_dict)

columns = ['imgPath', 'imgEncode', 'class', 'x', 'y', 'w', 'h']
# Create a panda DataFrame - no
gt_table = pd.DataFrame(data=rows_list, columns=columns)
gt_table.to_csv('gt_carBBox.csv', index=True)
print(gt_table.head(100))
