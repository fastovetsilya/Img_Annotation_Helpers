# Image annotation helpers
Bunch of tools to help annotate images

## Universal annotation converter
Command line tool to transform annotation formats between annotation apps. Both bounding box annotations for object detection (rectangles, circles etc.) and polygon annotations for instance segmentation (polygons) are covered. Labelme and VIA annotators are used for both polygon annotation for instance segmentation and annotation with rectangular bounding boxes for standard object detection. The polygons are transformed to bounding boxes to be used with YOLO using minimum covering rectangle. 

### Requirements
```pip install -r requirements.txt```

OR

```pip install numpy```

```pip install opencv_python```

### Available methods:

1) Convert Labelme rectangles to YOLO format
```python labelmerect_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```

2) Convert YOLO annotations to Labelme rectangles
```python yolo_to_labelmerect --input_dir=/path/to/annotations/ --output_dir=/output/path/```
To open annotations with Labelme, use ```labelme --nodata``` command

3) Convert Labelme polygons to YOLO bounding boxes 
```python labelmepoly_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```

4) Convert Labelme polygons to YOLOv4c ellipse coordinates (only circles supported now)
```python labelmepoly_to_yolov4c --input_dir=/path/to/annotations/ --output_dir=/output/path/```

5) Convert Labelme polygons to VIA image annotation format (only polygons supported now)
```python labelme_to_via --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Group IDs for Labelme can also be converted. By default, the methods looks for the attribute name "Group_ID" in VIA annotations. The default group ID attribute name can be changed by setting ```----group_id_name``` argument. For example, use: 
```python labelme_to_via --input_dir=/path/to/annotations/ --output_dir=/output/path/ --group_id_name=Track_ID``` to parse Track_ID attribute from VIA to "Group ID" column in Labelme 

6) Convert VIA polygos to Labelme image annotation format (only polygons supported now)
```python via_to_labelme --input_dir=/path/to/annotations/ --output_dir=/output/path/```. To open annotations with Labelme, use ```labelme --nodata``` command

7) Create empty annotation .txt files in YOLO format for images without annotations. For better performance of YOLO, you should add images with no annotations. This method creates empty annotation files for them
```python create_empty_txt_yolo --input_dir=/path/to/files/```

Access help with ```python convert.py -h```. Also read comments in the code. 

### Todo list: 

- [x] Add support for reverse VIA to Labelme converter (02/06/2021)

- [x] Add YOLO (LabelImg) to Labelme and Labelme to YOLO converters (02/10/2021)

- [ ] Add approximation of polygons with ellipses for YOLOv4c

- [ ] Add Labelme to VIA and VIA to Labelme converters for rectangles (bounding boxes)

- [ ] Add YOLO (LabelImg) to VIA and VIA to YOLO converters
