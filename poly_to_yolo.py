"""
Created on Mon Sep 21 12:13:11 2020

@author: fastovetsilya

Create rectangular bounding boxes from Labelme polygons 
and save in YOLO format (for YOLO and LabelImg). 
Uses extreme points from the polygons to create the bounding boxes.
"""

import shutil, glob 
import json
import numpy as np

#######################################################################
# Define input and output dir here
in_poly_dir = './test/poly_to_yolo/json_polygons/'
out_yolo_dir = './test/poly_to_yolo/yolo_rbbx/'
#######################################################################

# Create the list of .json files with polygon annotations
poly_file_list = glob.glob(in_poly_dir + '*.json')
# Initialize the list of labels
label_list = [] 

# Create the list of labels
for poly_file in poly_file_list:
    with open(poly_file, 'r') as f:
        poly_annotations = f.read()
    poly_annotations = json.loads(poly_annotations)
    poly_shapes = poly_annotations['shapes']
    
    for shape in poly_shapes:
        label = shape['label']
        if label not in label_list:
            label_list.append(label)

label_list = [label_list, [i for i in range(len(label_list))]]
       

#######################################
# Create YOLO annotations for polygon annotations

for poly_file in poly_file_list:
    
    # Load polygon annotation
    with open(poly_file, 'r') as f:
        poly_annotations = f.read()
    poly_annotations = json.loads(poly_annotations)
    poly_shapes = poly_annotations['shapes']
    img_dims = [poly_annotations['imageWidth'], poly_annotations['imageHeight']] # [width, height]
    
    # Create YOLO output file
    yolo_outfile_name = poly_file.split('/')[-1].replace('.json', '.txt')
    yolo_outfile = open(out_yolo_dir + yolo_outfile_name, 'w')
    yolo_outfile = open(out_yolo_dir + yolo_outfile_name, 'a')
    
    # Iterate through the shapes
    for shape in poly_shapes:
        shape_type = shape['shape_type']
        if shape_type != 'polygon':
            print('Warning! The annotations contain shapes other than polygons')
            continue
        shape_points = np.array(shape['points'])
        shape_label = shape['label']
        shape_label_id = label_list[0].index(shape_label)
        # Find extreme points
        box_width = (max(shape_points[:, 0]) - min(shape_points[:, 0]))
        box_height = (max(shape_points[:, 1]) - min(shape_points[:, 1]))
        box_center_x = min(shape_points[:, 0]) + box_width / 2
        box_center_y = min(shape_points[:, 1]) + box_height / 2
        # Create a box in YOLO format
        box_yolo = [box_center_x / img_dims[0],
                    box_center_y / img_dims[1], 
                    box_width / img_dims[0],
                    box_height / img_dims[1]]
        
        box_yolo_out = str([shape_label_id] + box_yolo)
        box_yolo_out = box_yolo_out.replace(',', '').replace('[', '').replace(']','')
        # Write box into file
        yolo_outfile.write(box_yolo_out)
        yolo_outfile.write('\n')
    
    # Close the file with YOLO annotaions 
    yolo_outfile.close()
    # Copy image to that file
    shutil.copyfile(poly_file.replace('.json', '.jpg'), 
                    out_yolo_dir + yolo_outfile_name.replace('.txt', '.jpg'))

# Create classes.txt
classes_file = open(out_yolo_dir + 'classes.txt', 'w')
classes_file = open(out_yolo_dir + 'classes.txt', 'a')

for i in range(np.shape(label_list)[1]):
    classes_file.write(str(label_list[1][i]) + ' ' + label_list[0][i])
    classes_file.write('\n')

classes_file.close()