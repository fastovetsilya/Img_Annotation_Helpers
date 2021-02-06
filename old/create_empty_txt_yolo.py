"""
Creates empty txt for YOLO for images without objects
"""
from glob import glob

# Define data directory
input_dir = 'fully_prepared/'

# Images list
img_list = glob(input_dir + '*.jpg')

# Txt list
txt_list = glob(input_dir + '*.txt')

# Go through images and create txt files if it does not exist
for img in img_list:
    txt_file = img.replace('.jpg', '.txt')
    
    if txt_file not in txt_list:
        file = open(txt_file, 'w')
        file.close()