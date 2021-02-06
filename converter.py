"""
Universal converter between image annotation formats. 
For the list of supported operations see README.md
"""
# Import libraries
import cv2
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


def labelmepoly2yolo(input_dir, output_dir, label_list):
    """
    Transform image annotations (polygons) from Labelme format to YOLO format
    Automatically generate bounding boxes from polygons
    
    # TODO: add support for other shapes
    
    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations
        
        ::output_dir: str, path to the directory to save VIA annotations
        
        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
    """
    
    poly_file_list = glob.glob(input_dir + "*.json")
    for poly_file in poly_file_list:
        # Load polygon annotation
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        image_format=".jpg" #Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except: 
            print("Warning! Format detection unsuccessful")
            pass
        img_dims = [poly_annotations["imageWidth"], poly_annotations["imageHeight"]] # [width, height]
        
        # Create YOLO output file
        yolo_outfile_name = poly_file.split("/")[-1].replace(".json", ".txt")
        yolo_outfile = open(output_dir + yolo_outfile_name, "w")
        yolo_outfile = open(output_dir + yolo_outfile_name, "a")
        
        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon":
                print("Warning! The annotations contain shapes other than polygons")
                continue
            shape_points = np.array(shape["points"])
            shape_label = shape["label"]
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
            box_yolo_out = box_yolo_out.replace(",", "").replace("[", "").replace("]","")
            # Write box into file
            yolo_outfile.write(box_yolo_out)
            yolo_outfile.write("\n")
        
        # Close the file with YOLO annotaions 
        yolo_outfile.close()
        # Copy image to that file
        shutil.copyfile(poly_file.replace(".json", image_format), 
                        output_dir + yolo_outfile_name.replace(".txt", image_format))
    
    # Create classes.txt
    classes_file = open(os.path.join(output_dir, "classes.txt"), "w")
    classes_file = open(os.path.join(output_dir, "classes.txt"), "a")
    
    for i in range(np.shape(label_list)[1]):
        classes_file.write(str(label_list[1][i]) + " " + label_list[0][i])
        classes_file.write("\n")
    
    classes_file.close()



def labelmepoly2yolov4c(input_dir, output_dir, label_list):
    """
    Transform image annotations (polygons) from Labelme format to YOLOv4 format
    Automatically approximate polygons with circles
    Automatically generate bounding boxes from polygons
    
    # TODO: add automatic approximation of polygons with ellipses
    # TODO: add support for other shapes
    
    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations
        
        ::output_dir: str, path to the directory to save VIA annotations
        
        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
    """
    
    poly_file_list = glob.glob(input_dir + "*.json")
    for poly_file in poly_file_list:
        # Load polygon annotation
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        image_format=".jpg" #Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except: 
            print("Warning! Format detection unsuccessful")
            pass
        img_dims = [poly_annotations["imageWidth"], poly_annotations["imageHeight"]] # [width, height]
        
        # Create YOLO output file
        yolo_outfile_name = poly_file.split("/")[-1].replace(".json", ".txt")
        yolo_outfile = open(output_dir + yolo_outfile_name, "w")
        yolo_outfile = open(output_dir + yolo_outfile_name, "a")
        
        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon":
                print("Warning! The annotations contain shapes other than polygons")
                continue
            shape_points = np.array(shape["points"])
            shape_contours = np.expand_dims(shape_points, 1).astype(np.int32)
            shape_label = shape["label"]
            shape_label_id = label_list[0].index(shape_label)
            # Fit a circle to the poligon
            fitted_circle = cv2.minEnclosingCircle(shape_contours)
            # Find extreme points for box
            box_width = fitted_circle[1] * 2
            box_height = fitted_circle[1] * 2 
            box_center_x = fitted_circle[0][0] 
            box_center_y = fitted_circle[0][1]
            # Create a box in YOLO format
            box_yolo = [box_center_x / img_dims[0],
                        box_center_y / img_dims[1], 
                        box_width / img_dims[0],
                        box_height / img_dims[1]]
            
            box_yolo_out = str([shape_label_id] + box_yolo)
            box_yolo_out = box_yolo_out.replace(",", "").replace("[", "").replace("]","")
            # Write box into file
            yolo_outfile.write(box_yolo_out)
            yolo_outfile.write("\n")
        
        # Close the file with YOLO annotaions 
        yolo_outfile.close()
        # Copy image to that file
        shutil.copyfile(poly_file.replace(".json", image_format), 
                        output_dir + yolo_outfile_name.replace(".txt", image_format))
    
    # Create classes.txt
    classes_file = open(output_dir + "classes.txt", "w")
    classes_file = open(output_dir + "classes.txt", "a")
    
    for i in range(np.shape(label_list)[1]):
        classes_file.write(str(label_list[1][i]) + " " + label_list[0][i])
        classes_file.write("\n")
    
    classes_file.close()


def labelme2via(input_dir, output_dir, label_list):
    """
    Transform image annotations from Labelme to VIA .json annotations format
    
    # TODO: add support for other shapes
    
    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations
        
        ::output_dir: str, path to the directory to save VIA annotations
        
        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
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
        # img_dims = [poly_annotations["imageWidth"], poly_annotations["imageHeight"]] # [width, height]
        image_format=".jpg" #Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except: 
            print("Warning! Format detection unsuccessful")
            pass
        img_size = int(os.stat(os.path.join(poly_file.replace(".json", image_format))).st_size)
        
        # Initialize regions list (VIA format) and terate through the shapes
        regions = [] 
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon": 
                print("Non-polygon shapes not supported. For support, modify labelme2via()")
                break
            shape_points = shape["points"]
            shape_label = shape["label"]
            #shape_label_id = label_list[0].index(shape_label)
            
            # Create region part
            region = {}
            region["region_attributes"] = {}
            region["shape_attributes"] = {}
            region["shape_attributes"]["name"] = "polygon"
            all_points_x = [int(point[0]) for point in shape_points]
            all_points_y = [int(point[1]) for point in shape_points]
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
        annotations[str(image_name) + str(img_size)] = ann
        
    # Save .json file with VIA annotations
    annotations = json.dumps(annotations)
    with open(os.path.join(output_dir, "via_annotations.json"), "w") as outfile:  
        outfile.write(annotations)
 

if __name__=="__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Universal annotation converter")
    
    parser.add_argument("command", metavar="<command>", 
                        help="labelme_to_via")
    parser.add_argument("--input_dir", required=False,
                        metavar="input/path", 
                        default="./test/poly_to_yolov4c/json_polygons/",
                        help="Provide input directory with Labelme annotations")
    parser.add_argument("--output_dir", required=False,
                        metavar="output/path", 
                        default="./test/poly_to_yolov4c/yolo_cbbx/", 
                        help="Provide output directory with VIA annotations")
    
    args = parser.parse_args()
    
    # Get label list
    labels = extract_labelme_labels(args.input_dir) 
    
    # Perform transformation from Labelme to VIA
    if args.command == "labelmepoly_to_yolo":
        labelmepoly2yolo(args.input_dir, args.output_dir, labels)
    elif args.command == "labelmepoly_to_yolov4c":
        labelmepoly2yolov4c(args.input_dir, args.output_dir, labels)
    elif args.command == "labelme_to_via":
        labelme2via(args.input_dir, args.output_dir, labels)
    else: 
        print("Error: command not recognized. Stop.")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    