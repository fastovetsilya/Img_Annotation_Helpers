"""
Universal converter between image annotation formats. 
For the list of supported operations see README.md
"""
# Import libraries
import json
import shutil
import numpy as np
from glob import glob


def extract_labelme_labels(labelme_input_directory):
    """
    Extracts the list of labels from .json Labelme annotations and creates
    the list of label ids for them (to use with YOLO). 
    
    Parameter: 
        ::labelme_input_directory: str, directory with Labelme .json 
        annotations. 
    """    
    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(labelme_input_directory + '*.json')
    # Initialize the list of labels
    label_list = [] 
    
    # Create the list of labels
    # Iterate files
    for poly_file in poly_file_list:
        with open(poly_file, 'r') as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations['shapes']
        
        # Iterate shapes in each file
        for shape in poly_shapes:
            label = shape['label']
            if label not in label_list:
                label_list.append(label)
    
    # Sort labels (for consistency)
    label_list.sort()
    # Return labels as list of list of char labels and int label ids
    label_list = [label_list, [i for i in range(len(label_list))]]
    
    return label_list


def labelme2via(input_dir, output_dir, label_list):
    """
    Transform image annotations from Labelme to VIA .json annotations format
    
    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations
        
        ::output_dir: str, path to the directory to save VIA annotations
        
        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation.
    """
    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(input_dir + '*.json')
    for poly_file in poly_file_list:
        
        # Load polygon annotation
        with open(poly_file, 'r') as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations['shapes']
        img_dims = [poly_annotations['imageWidth'], poly_annotations['imageHeight']] # [width, height]
        img_size = img_dims[0] * img_dims[1]
        
        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape['shape_type']
            if shape_type != 'polygon': # TODO: add support for other shapes
                print('Non-polygon shapes not supported. For support, modify labelme2via()')
                break
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


 

if __name__=="__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Labelme to VIA converter')
    
    parser.add_argument("command", metavar="<command>", 
                        help="convert")
    parser.add_argument("--input_dir", required=True,
                        metavar="/path/to/labelme/annotations/", 
                        default="./test/lme_to_via/json_polygons/",
                        help="Provide input directory with Labelme annotations")
    parser.add_argument("--output_dir", required=True,
                        metavar="/path/to/via/annotations/", 
                        default="./test/lme_to_via/via_polygons", 
                        help="Provide output directory with VIA annotations")
    
    args = parser.parse_args()
    
    # Get label list
    labels = extract_labelme_labels(args.input_dir) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    