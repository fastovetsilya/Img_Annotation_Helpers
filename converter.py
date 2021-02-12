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
    Extract the list of labels from .json Labelme annotations and create
    the list of label ids for them (to use with YOLO). 

    Parameter: 
        ::labelme_input_directory: str, directory with Labelme .json 
        annotations. 
    """
    # Check the directory name and correct if needed
    if labelme_input_directory[-1] != "/":
        print("Correcting path: adding '/' to the end")
        labelme_input_directory += "/"

    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(os.path.join(labelme_input_directory, "*.json"))
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


def create_empty_txt_yolo(input_dir, image_format=".jpg"):
    """
    Create empty txt for YOLO for images without objects (annotations)
    Apply to the directory with YOLO annotations 
    Set input_dir == output_dir to save in the same directory
    TODO: add automatic recognition of image format

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations. The 
        files are created in the same directory (to complete YOLO dataset)

        ::image_format: str, format of the images. Default is ".jpg"
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"

    # Get images list
    img_list = glob.glob(os.path.join(input_dir, image_format))
    # Txt list
    txt_list = glob(input_dir + "*.txt")
    # Go through images and create txt files if it does not exist
    for img in img_list:
        txt_file = img.replace(image_format, ".txt")
        if txt_file not in txt_list:
            file = open(txt_file, "w")
            file.close()


def labelmerect2yolo(input_dir, output_dir, label_list):
    """
    Transform image annotations (rectangles) from Labelme format to YOLO format
    Transform rectangles only and skip other shapes

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations

        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    poly_file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for poly_file in poly_file_list:
        # Load polygon annotation
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        image_format = ".jpg"  # Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except:
            print("Warning! Format detection unsuccessful")
            pass
        img_dims = [poly_annotations["imageWidth"],
                    poly_annotations["imageHeight"]]  # [width, height]

        # Create YOLO output file
        yolo_outfile_name = poly_file.split("/")[-1].replace(".json", ".txt")
        yolo_outfile = open(output_dir + yolo_outfile_name, "w")
        yolo_outfile = open(output_dir + yolo_outfile_name, "a")

        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "rectangle":
                print("Warning! The annotations contain shapes other than rectangles")
                print("This method will only convert rectangles to YOLO")
                continue
            shape_points = np.array(shape["points"])
            shape_label = shape["label"]
            shape_label_id = label_list[0].index(shape_label)
            # Find extreme points
            box_width = shape_points[1, 0] - shape_points[0, 0]
            box_height = shape_points[0, 1] - shape_points[1, 1]
            box_center_x = shape_points[0, 0] + box_width / 2
            box_center_y = shape_points[1, 1] + box_height / 2
            # Create a box in YOLO format
            box_yolo = [box_center_x / img_dims[0],
                        box_center_y / img_dims[1],
                        abs(box_width / img_dims[0]),
                        abs(box_height / img_dims[1])]
            box_yolo_out = str([shape_label_id] + box_yolo)
            box_yolo_out = box_yolo_out.replace(
                ",", "").replace("[", "").replace("]", "")
            # Write box into file
            yolo_outfile.write(box_yolo_out)
            yolo_outfile.write("\n")

        # Close the file with YOLO annotaions
        yolo_outfile.close()
        # Copy image to that file
        shutil.copyfile(poly_file.replace(".json", image_format),
                        os.path.join(output_dir, yolo_outfile_name.replace(".txt", image_format)))

    # Create classes.txt
    classes_file = open(os.path.join(output_dir + "classes.txt"), "w")
    classes_file = open(os.path.join(output_dir + "classes.txt"), "a")

    for i in range(np.shape(label_list)[1]):
        classes_file.write(str(label_list[1][i]) + " " + label_list[0][i])
        classes_file.write("\n")

    classes_file.close()


def yolo2labelmerect(input_dir, output_dir):
    """
    Transform YOLO image annotations (rectangles) to Labelme format

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations
    """

    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    # Get classes list
    label_list = [[], []]
    f = open(os.path.join(input_dir, "classes.txt"), "r")
    while True:
        newline = f.readline()
        if not newline.replace("\n", ""):
            break
        label_list[0].append(newline.split(" ")[1].replace("\n", ""))
        label_list[1].append(int(newline.split(" ")[0]))
    f.close()
    if len(label_list[0]) != len(label_list[1]):
        print("Error: label list integrity check failed. Please check classes.txt")
        return 1

    # Get annotations and images list
    poly_file_list = glob.glob(os.path.join(input_dir, "*.txt"))
    images_list = [fname for fname in glob.glob(
        os.path.join(input_dir, "*")) if fname not in poly_file_list]
    poly_file_list = [fname for fname in poly_file_list if fname.split(
        "/")[-1] != "classes.txt"]

    # Automatically find image format and check if format is the same for all images
    image_format = [file_ext.split(".")[-1] for file_ext in images_list]
    if not all(file_ext == image_format[0] for file_ext in image_format):
        print("Error: not all images in the direcory have the same format. Please check the images")
        return 1
    image_format = "." + image_format[0]

    # Convert YOLO annotations to Labelme format
    for poly_file in poly_file_list:
        # Load image
        image_filename = poly_file.replace(".txt", image_format).split("/")[-1]
        try:
            image = cv2.imread(os.path.join(input_dir, image_filename))
        except:
            print("Warning: image {} could not be loaded. Continue... ".format(
                image_filename))
            continue

        # Create and fill labelme dictionary with necessary data
        labelme_ann = {}
        labelme_ann["version"] = "4.5.6"  # Add any other version
        labelme_ann["flags"] = {}
        labelme_ann["fillColor"] = [int(i) for i in list(
            np.random.randint(255, size=3))]  # Generate random color
        labelme_ann["lineColor"] = [
            int(i) for i in list(np.random.randint(255, size=3))]
        labelme_ann["imageData"] = None
        labelme_ann["imageHeight"] = image.shape[0]
        labelme_ann["imageWidth"] = image.shape[1]
        labelme_ann["imagePath"] = image_filename

        # Create shapes
        shapes = []
        # Read file with YOLO annotations
        f = open(poly_file, "r")
        while True:
            poly_annotation = f.readline()
            if not poly_annotation:
                f.close()
                break

            # Extract data from annotaion
            poly_annotation = poly_annotation.replace("\n", "")
            poly_annotation = poly_annotation.split(" ")
            poly_annotation = [float(ann) for ann in poly_annotation]
            label_id = int(poly_annotation[0])
            label = label_list[0][label_list[1].index(label_id)]

            # YOLO format: <center x>, <center y>, width, height (scaled)
            yolo_ann = [poly_annotation[1], poly_annotation[2],
                        poly_annotation[3], poly_annotation[4]]

            # Rescale yolo annotation (according to image size)
            yolo_ann = [yolo_ann[0] * image.shape[1],
                        yolo_ann[1] * image.shape[0],
                        yolo_ann[2] * image.shape[1],
                        yolo_ann[3] * image.shape[0]]

            # Create shape points
            points = [[yolo_ann[0] - yolo_ann[2] / 2, yolo_ann[1] + yolo_ann[3] / 2],
                      [yolo_ann[0] + yolo_ann[2] / 2, yolo_ann[1] - yolo_ann[3] / 2]]

            # Create shape
            shape = {}
            shape["line_color"] = None
            shape["fill_color"] = None
            shape["label"] = label
            shape["points"] = points
            shape["group_id"] = None
            shape["shape_type"] = "rectangle"
            shape["flags"] = {}
            shapes.append(shape)
        f.close()
        labelme_ann["shapes"] = shapes

        # Write annotations to the file
        labelme_ann = json.dumps(labelme_ann)
        lme_ann_name = image_filename.replace(
            image_filename.split(".")[-1], "json")

        with open(os.path.join(output_dir, lme_ann_name), "w") as f:
            f.write(labelme_ann)

        # Copy image file
        shutil.copyfile(os.path.join(input_dir, image_filename),
                        os.path.join(output_dir, image_filename))


def labelmepoly2yolo(input_dir, output_dir, label_list):
    """
    Transform image annotations (polygons) from Labelme format to YOLO format
    Automatically generate bounding boxes from polygons. 
    Transform polygons only and skip other shapes 

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations

        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    poly_file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for poly_file in poly_file_list:
        # Load polygon annotation
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        image_format = ".jpg"  # Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except:
            print("Warning! Format detection unsuccessful")
            pass
        img_dims = [poly_annotations["imageWidth"],
                    poly_annotations["imageHeight"]]  # [width, height]

        # Create YOLO output file
        yolo_outfile_name = poly_file.split("/")[-1].replace(".json", ".txt")
        yolo_outfile = open(output_dir + yolo_outfile_name, "w")
        yolo_outfile = open(output_dir + yolo_outfile_name, "a")

        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon":
                print("Warning! The annotations contain shapes other than polygons")
                print("This method will convert only polygons to YOLO")
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
            box_yolo_out = box_yolo_out.replace(
                ",", "").replace("[", "").replace("]", "")
            # Write box into file
            yolo_outfile.write(box_yolo_out)
            yolo_outfile.write("\n")

        # Close the file with YOLO annotaions
        yolo_outfile.close()
        # Copy image to that file
        shutil.copyfile(poly_file.replace(".json", image_format),
                        os.path.join(output_dir, yolo_outfile_name.replace(".txt", image_format)))

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

    # TODO: add automatic approximation of polygons with ellipses instead 
    of circles

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations

        ::label_list: list, structure of type [[labels], [label_ids]]. E.g.
        [[Apples, Oranges], [0, 1]]. Use extract_labelme_labels() for 
        automatic generation
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    poly_file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for poly_file in poly_file_list:
        # Load polygon annotation
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        image_format = ".jpg"  # Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except:
            print("Warning! Format detection unsuccessful")
            pass
        img_dims = [poly_annotations["imageWidth"],
                    poly_annotations["imageHeight"]]  # [width, height]

        # Create YOLO output file
        yolo_outfile_name = poly_file.split("/")[-1].replace(".json", ".txt")
        yolo_outfile = open(output_dir + yolo_outfile_name, "w")
        yolo_outfile = open(output_dir + yolo_outfile_name, "a")

        # Iterate through the shapes
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type != "polygon":
                print("Warning! The annotations contain shapes other than polygons")
                print("This method will convert only polygons to YOLO")
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
            box_yolo_out = box_yolo_out.replace(
                ",", "").replace("[", "").replace("]", "")
            # Write box into file
            yolo_outfile.write(box_yolo_out)
            yolo_outfile.write("\n")

        # Close the file with YOLO annotaions
        yolo_outfile.close()
        # Copy image to that file
        shutil.copyfile(poly_file.replace(".json", image_format),
                        os.path.join(output_dir + yolo_outfile_name.replace(".txt", image_format)))

    # Create classes.txt
    classes_file = open(os.path.join(output_dir, "classes.txt"), "w")
    classes_file = open(os.path.join(output_dir, "classes.txt"), "a")

    for i in range(np.shape(label_list)[1]):
        classes_file.write(str(label_list[1][i]) + " " + label_list[0][i])
        classes_file.write("\n")

    classes_file.close()


def labelme2via(input_dir, output_dir, groupid_name="Group_ID"):
    """
    Transform image annotations from Labelme to VIA .json annotations format

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    # Initialize VIA annotaions
    annotations = {}

    # Create the list of .json files with polygon annotations
    poly_file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for poly_file in poly_file_list:

        # Load the files with annotations and extract the necessary data
        with open(poly_file, "r") as f:
            poly_annotations = f.read()
        poly_annotations = json.loads(poly_annotations)
        poly_shapes = poly_annotations["shapes"]
        # img_dims = [poly_annotations["imageWidth"], poly_annotations["imageHeight"]] # [width, height]
        image_format = ".jpg"  # Default file format
        # Try automatic file format detection
        try:
            image_format = "." + poly_annotations["imagePath"].split(".")[-1]
        except:
            print("Warning! Format detection unsuccessful")
            pass
        img_size = int(os.stat(os.path.join(
            poly_file.replace(".json", image_format))).st_size)

        # Initialize regions list (VIA format) and terate through the shapes
        regions = []
        for shape in poly_shapes:
            shape_type = shape["shape_type"]
            if shape_type not in ["polygon", "rectangle"]:
                print(
                    "Error: shapes other than polygons and rectangles not yet supported. Please modify labelme2via()")
                return 1
            shape_points = shape["points"]
            shape_label = shape["label"]
            #shape_label_id = label_list[0].index(shape_label)
            shape_groupid = shape["group_id"]

            # Create region part
            region = {}
            region["region_attributes"] = {}
            region["shape_attributes"] = {}

            # If shape is a polygon:
            if shape_type == "polygon":
                region["shape_attributes"]["name"] = "polygon"
                all_points_x = [int(point[0]) for point in shape_points]
                all_points_y = [int(point[1]) for point in shape_points]
                region["shape_attributes"]["all_points_x"] = all_points_x
                region["shape_attributes"]["all_points_y"] = all_points_y
            # If shape is a rectangle
            if shape_type == "rectangle":
                region["shape_attributes"]["name"] = "rect"
                x = int(shape_points[0][0])
                y = int(shape_points[1][1])
                width = int(shape_points[1][0] - shape_points[0][0])
                height = int(shape_points[0][1] - shape_points[1][1])
                region["shape_attributes"]["x"] = x
                region["shape_attributes"]["y"] = y
                region["shape_attributes"]["width"] = width
                region["shape_attributes"]["height"] = height

            region["region_attributes"]["label"] = shape_label
            if shape_groupid:
                region["region_attributes"][groupid_name] = shape_groupid

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

        # Copy image file
        image_filename = poly_file.split(
            "/")[-1].replace(".json", image_format)
        shutil.copyfile(os.path.join(input_dir, image_filename),
                        os.path.join(output_dir, image_filename))

    # Save .json file with VIA annotations
    annotations = json.dumps(annotations)
    with open(os.path.join(output_dir, "via_annotations.json"), "w") as outfile:
        outfile.write(annotations)


def via2labelme(input_dir, output_dir, groupid_name="Group_ID"):
    """
    Transform image annotations from VIA to Labelme .json annotations format

    Parameters:
        ::input_dir: str, path to the directory with Labelme annotations

        ::output_dir: str, path to the directory to save annotations

        ::groupid_attr: str, group id attribute name. Default is None (not used)
    """
    # Check the directory name and correct if needed
    if input_dir[-1] != "/":
        print("Correcting input path: adding '/' to the end")
        input_dir += "/"
    if output_dir[-1] != "/":
        print("Correcting output path: adding '/' to the end")
        output_dir += "/"

    # Find file with annotations
    poly_file = glob.glob(os.path.join(input_dir, "*.json"))
    # Check if file exists and if it is the only .json file in the directory
    if poly_file == []:
        print("Error. No annotation file found. Check that .json annotations are in the input directory")
        return 1
    if len(poly_file) > 1:
        print("Error. There are more than one .json files in the input directory. Check VIA annotations")
        return 1
    poly_file = poly_file[0]

    # Load annotaions
    with open(poly_file, "r") as f:
        via_annotations = f.read()
    via_annotations = json.loads(via_annotations)
    via_annotations = list(via_annotations.values())

    # Create labelme annotation file for each via annotations entry
    for via_annotation in via_annotations:
        file_name = via_annotation["filename"]
        # Open image to find its size
        image = cv2.imread(os.path.join(input_dir, file_name))

        # Create and fill labelme dictionary with necessary data
        labelme_ann = {}
        labelme_ann["version"] = "4.5.6"  # Add any other version
        labelme_ann["flags"] = {}
        labelme_ann["fillColor"] = [int(i) for i in list(
            np.random.randint(255, size=3))]  # Generate random color
        labelme_ann["lineColor"] = [
            int(i) for i in list(np.random.randint(255, size=3))]
        labelme_ann["imageData"] = None
        labelme_ann["imageHeight"] = image.shape[0]
        labelme_ann["imageWidth"] = image.shape[1]
        labelme_ann["imagePath"] = file_name
        # Create shapes
        shapes = []
        for region in via_annotation["regions"]:
            label = region["region_attributes"]['label']
            # Group ID attribute
            if groupid_name not in region["region_attributes"]:
                group_id = None
            else:
                try:
                    group_id = int(region["region_attributes"][groupid_name])
                except:
                    print("Warning: could not convert one or more group IDs in {}. Continue...".format(
                        file_name))
                    group_id = None

            shape_type = region["shape_attributes"]["name"]
            if shape_type not in ["polygon", "rect"]:
                print(
                    "Error: shapes other than polygons and rectangles not yet supported. Please modify via2labelme()")
                return 1

            # Initialize shape
            shape = {}
            # If shape type is a polygon
            if shape_type == "polygon":
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]
                if len(all_points_x) != len(all_points_y):
                    print("Error: annotation points mismatch")
                    return 1
                points = []
                for i in range(len(all_points_x)):
                    points.append([int(all_points_x[i]), int(all_points_y[i])])
                shape["shape_type"] = "polygon"

            # If shape is a rectangle
            if shape_type == "rect":
                x = region["shape_attributes"]["x"]
                y = region["shape_attributes"]["y"]
                width = region["shape_attributes"]["width"]
                height = region["shape_attributes"]["height"]
                shape["shape_type"] = "rectangle"
                points = [[x, y], [x + width, y + height]]

            # Create shape
            shape["line_color"] = None
            shape["fill_color"] = None
            shape["label"] = label
            shape["points"] = points
            shape["group_id"] = group_id
            shape["flags"] = {}
            shapes.append(shape)

        labelme_ann["shapes"] = shapes

        # Write annotations to the file
        labelme_ann = json.dumps(labelme_ann)
        lme_ann_name = file_name.replace(file_name.split(".")[-1], "json")
        with open(os.path.join(output_dir, lme_ann_name), "w") as f:
            f.write(labelme_ann)

        # Copy image file
        shutil.copyfile(os.path.join(input_dir, file_name),
                        os.path.join(output_dir, file_name))


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Universal annotation converter")
    parser.add_argument("command", metavar="<command>",
                        help="Choose the method. See README.md for the list of available methods")
    parser.add_argument("--input_dir", required=True,
                        metavar="input/path",
                        help="Provide input directory")
    parser.add_argument("--output_dir", required=False,
                        metavar="output/path",
                        help="Provide output directory. Output directory IS NOT created automatically")
    parser.add_argument("--group_id_name", required=False,
                        default="Group_ID",
                        help="Provide the name of the attribute to parse in the Group ID column (optional)")
    args = parser.parse_args()

    # Check input directory
    if glob.glob(args.input_dir + "*.json") == []:
        print("Warning: json check found no .json files found in the input directory: {}".format(
            args.input_dir))

    # Perform transformation from Labelme to VIA
    if args.command == "labelmepoly_to_yolo":
        if args.output_dir is not None:
            labels = extract_labelme_labels(args.input_dir)
            labelmepoly2yolo(args.input_dir, args.output_dir, labels)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "labelmerect_to_yolo":
        if args.output_dir is not None:
            labels = extract_labelme_labels(args.input_dir)
            labelmerect2yolo(args.input_dir, args.output_dir, labels)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "yolo_to_labelmerect":
        if args.output_dir is not None:
            yolo2labelmerect(args.input_dir, args.output_dir)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "labelmepoly_to_yolov4c":
        if args.output_dir is not None:
            labels = extract_labelme_labels(args.input_dir)
            labelmepoly2yolov4c(args.input_dir, args.output_dir, labels)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "labelme_to_via":
        if args.output_dir is not None:
            labelme2via(args.input_dir, args.output_dir, args.group_id_name)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "via_to_labelme":
        if args.output_dir is not None:
            via2labelme(args.input_dir, args.output_dir, args.group_id_name)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "viapoly_to_yolo":
        if args.output_dir is not None:
            # Convert to Labelme annotations and then Labelme to YOLO
            tmp_dir = os.path.join(args.output_dir, "tmp/")
            os.mkdir(tmp_dir)
            via2labelme(args.input_dir, tmp_dir, args.group_id_name)
            labels = extract_labelme_labels(tmp_dir)
            labelmepoly2yolo(tmp_dir, args.output_dir, labels)
            shutil.rmtree(tmp_dir)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "viarect_to_yolo":
        if args.output_dir is not None:
            # Convert to Labelme annotations and then Labelme to YOLO
            tmp_dir = os.path.join(args.output_dir, "tmp/")
            os.mkdir(tmp_dir)
            via2labelme(args.input_dir, tmp_dir, args.group_id_name)
            labels = extract_labelme_labels(tmp_dir)
            labelmerect2yolo(tmp_dir, args.output_dir, labels)
            shutil.rmtree(tmp_dir)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "yolo_to_viarect":
        if args.output_dir is not None:
            # Convert to Labelme annotations and then Labelme to YOLO
            tmp_dir = os.path.join(args.output_dir, "tmp/")
            os.mkdir(tmp_dir)
            yolo2labelmerect(args.input_dir, tmp_dir)
            labels = extract_labelme_labels(tmp_dir)
            labelme2via(tmp_dir, args.output_dir, labels)
            shutil.rmtree(tmp_dir)
        else:
            print("Error: parameter output_dir is required for this method.")
            pass
    elif args.command == "create_empty_txt_yolo":
        create_empty_txt_yolo(args.input_dir)
    else:
        print("Error: command '{}' not recognized. Stop.".format(args.command))
