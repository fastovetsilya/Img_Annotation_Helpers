"""
Universal converter between image annotation formats. 
For the list of supported operations see README.md
"""
# Import libraries
import os
import json
import shutil
import glob
import numpy as np


def extract_labelme_labels(labelme_input_directory):
    """
    Extracts the list of labels from .json Labelme annotations and creates
    the list of label ids for them (to use with YOLO). 
    
    Parameter: 
        ::labelme_input_directory: str, directory with Labelme .json 
        annotations. 
    """    
    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(labelme_input_directory + "*.json")
    # Initialize the list of labels
    label_list = [] 
    
    # Create the list of labels
    # Iterate files
    for poly_file in poly_file_list:
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        
        # Iterate shapes in each file
        for shape in poly_shapes:
            label = shape["label"]
            if label not in label_list:
                label_list.append(label)
    
    # Sort labels (for consistency)
    label_list.sort()
    # Return labels as list of list of char labels and int label ids
    label_list = [label_list, [i for i in range(len(label_list))]]
    
    return label_list


def labelme2via(input_dir, output_dir, label_list, image_format=".jpg"):
    """
    Transform image annotations from Labelme to VIA .json annotations format
    
    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations
        
        ::output_dir: str, path to the directory to save VIA annotations
        
        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
        
        ::image_format: str, format of the images
        TODO: make it automatic
    """
    # Initialize VIA annotaions
    annotations = {}
    
    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(input_dir + "*.json")
    for poly_file in poly_file_list:
        
        # Load the files with annotations and extract the necessary data
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        img_dims = [poly_annotations["imageWidth"], poly_annotations["imageHeight"]] # [width, height]
        img_size = img_dims[0] * img_dims[1]
        
        # Initialize regions list (VIA format) and terate through the shapes
        regions = [] 
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon": # TODO: add support for other shapes
                print("Non-polygon shapes not supported. For support, modify labelme2via()")
                break
            shape_points = np.array(shape["points"])
            shape_label = shape["label"]
            #shape_label_id = label_list[0].index(shape_label)
            
            # Create region part
            region = {}
            region["region_attributes"] = {}
            region["shape_attributes"] = {}
            region["shape_attributes"]["name"] = "polygon"
            all_points_x = [point[0] for point in shape_points]
            all_points_y = [point[1] for point in shape_points]
            region["shape_attributes"]["all_points_x"] = all_points_x
            region["shape_attributes"]["all_points_y"] = all_points_y
            region["region_attributes"]["label"] = shape_label
            
            # Append to the list of regions
            regions.append(region)
        
        # Create annotation
        ann = {}
        ann["file_attributes"] = {}
        image_name = poly_file.split("/")[-1].replace(".json", image_format)
        ann["filename"] = image_name
        ann["regions"] = regions
        ann["size"] = img_size
        # Add annotation to the annotation list
        annotations[image_name + str(img_size)] = ann
        
    # Save .json file with VIA annotations
    annotations = json.dumps(annotations)
    with open(os.path.join(output_dir, "via_annotations.json"), "w") as outfile:  
        outfile.write(annotations)
 

if __name__=="__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Labelme to VIA converter")
    
    parser.add_argument("command", metavar="<command>", 
                        help="convert")
    parser.add_argument("--input_dir", required=False,
                        metavar="/path/to/labelme/annotations/", 
                        default="./test/lme_to_via/json_polygons/",
                        help="Provide input directory with Labelme annotations")
    parser.add_argument("--output_dir", required=False,
                        metavar="/path/to/via/annotations/", 
                        default="./test/lme_to_via/via_polygons", 
                        help="Provide output directory with VIA annotations")
    
    args = parser.parse_args()
    
    # Get label list
    labels = extract_labelme_labels(args.input_dir) 
    
    # Perform transformation from Labelme to VIA
    labelme2via(args.input_dir, args.output_dir, labels)
    
    # TODO: write parts of code for all the methods here
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    