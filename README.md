# Image annotation helpers
Bunch of tools to help annotate images

## Universal annotation converter
Command line tool to transform annotation formats between annotation apps. Both bounding box annotations for object detection (rectangles, circles etc.) and polygon annotations for instance segmentation (polygons) are covered. Labelme and VIA annotators are used for both polygon annotation for instance segmentation and annotation with rectangular bounding boxes for standard object detection. The polygons are transformed to bounding boxes to be used with YOLO using minimum covering rectangle. 
Currently supports convertation between the following apps: **Labelme**, **VGG image annotator (VIA)**, **LabelImg** (YOLO format).

### Requirements
```pip install -r requirements.txt```

OR

```pip install numpy```

```pip install opencv_python```

### Available methods:

1) Convert **Labelme rectangles to YOLO** format
```python converter.py labelmerect_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```

2) Convert **YOLO annotations to Labelme rectangles**
```python converter.py yolo_to_labelmerect --input_dir=/path/to/annotations/ --output_dir=/output/path/```
To open annotations with Labelme, use ```labelme --nodata``` command

3) Convert **Labelme polygons to YOLO bounding boxes** 
```python converter.py labelmepoly_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```
The polygons are transformed to bounding boxes to be used with YOLO using minimum covering rectangle

4) Convert **Labelme polygons to YOLOv4c** ellipse coordinates **(only circles supported now)**
```python converter.py labelmepoly_to_yolov4c --input_dir=/path/to/annotations/ --output_dir=/output/path/```

5) Convert **Labelme shapes to VIA** image annotation format **(only polygons and rectangles are supported now)**
```python converter.py labelme_to_via --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Group IDs can also be converted. By default, the method saves "Group ID" column in Labelme as "Group_ID" attribute in VIA annotations. The default group ID attribute name can be changed by setting ```----group_id_name``` argument. For example, use: 
```python converter.py labelme_to_via --input_dir=/path/to/annotations/ --output_dir=/output/path/ --group_id_name=Track_ID``` to save "Group ID" column in Labelme as Track_ID attribute in VIA

6) Convert **VIA shapes to Labelme** image annotation format **(only polygons and rectangles supported now)**
```python converter.py via_to_labelme --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Group IDs for Labelme can also be converted. By default, the method looks for the attribute name "Group_ID" in VIA annotations. The default group ID attribute name can be changed by setting ```----group_id_name``` argument. For example, use: 
```python converter.py via_to_labelme --input_dir=/path/to/annotations/ --output_dir=/output/path/ --group_id_name=Track_ID``` to parse Track_ID attribute from VIA to "Group ID" column in Labelme.
To open annotations with Labelme, use ```labelme --nodata``` command

7) Convert **VIA polygons to YOLO bounding boxes**
```python converter.py viapoly_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Indirect converter from VIA polygons to YOLO bounding boxes. Converts VIA polygons to Labelme polygons, and then converts Labelme polygons to YOLO by creating and removing /tmp directory in the output path.  
The polygons are transformed to bounding boxes to be used with YOLO using minimum covering rectangle.

8) Convert **VIA rectangles to YOLO bounding boxes**
```python converter.py viarect_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Indirect converter from VIA rectangles to YOLO bounding boxes. Converts VIA polygons to Labelme rectangles, and then converts Labelme rectangles to YOLO by creating and removing /tmp directory in the output path.  

9) Convert **YOLO bounding boxes to VIA rectangles**
```python converter.py yolo_to_viarect --input_dir=/path/to/annotations/ --output_dir=/output/path/```
Indirect converter from YOLO bounding boxes to VIA rectangles. Converts YOLO bounding boxes to Labelme rectangles, and then converts Labelme rectangles to VIA by creating and removing /tmp directory in the output path.  

10) Create **empty annotation .txt files in YOLO format** for images without annotations. For better performance of YOLO, you should add images with no annotations. This method creates empty annotation files for them
```python converter.py create_empty_txt_yolo --input_dir=/path/to/files/```

Access help with ```python converter.py -h```. Also read comments in the code. 

### Todo list: 

- [x] Add support for reverse VIA to Labelme converter (02/06/2021)

- [x] Add YOLO (LabelImg) to Labelme and Labelme to YOLO converters (02/10/2021)

- [x] Add YOLO (LabelImg) to VIA and VIA to YOLO converters(02/10/2021)

- [x] Add support for rectangle shapes for VIA-Labelme and Labelme-VIA converters

- [ ] Add Labelme to VIA and VIA to Labelme converters for other shapes (only polygons and rectangles supported now)

- [ ] Add approximation of polygons with ellipses for YOLOv4c
