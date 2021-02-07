# Image annotation helpers
Bunch of tools to help annotate images

## Universal annotaion converter
Command line tool to transform annotation formats between annotation apps

### Requirements
```pip install -r requirements.txt```

OR

```pip install numpy```

```pip install opencv_python```

### Available methods:

1) Convert Labelme rectangles to YOLO format: 
```python labelmerect_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```

2) Convert Labelme polygons to YOLO bounding boxes: 
```python labelmepoly_to_yolo --input_dir=/path/to/annotations/ --output_dir=/output/path/```

3) Convert Labelme polygons to YOLOv4c ellipse coordinates (only circles supported now)
```python labelmepoly_to_yolov4c --input_dir=/path/to/annotations/ --output_dir=/output/path/```

4) Convert Labelme polygons to VIA image annotation format (only polygons supported now)
```python labelme_to_via --input_dir=/path/to/annotations/ --output_dir=/output/path/```

5) Create empty annotation .txt files in YOLO format for images without annotations. For better performance of YOLO, you should add images with no annotations. This method creates empty annotation files for them. 
```python create_empty_txt_yolo --input_dir=/path/to/files/```

Access help with:```python convert.py -h```. Also read comments in the code. 

### Todo list: 

- [ ] Add approximation of polygons with ellipses for YOLOv4c

- [ ] Add Labelme to VIA converter for rectangles

- [ ] Add support for reverse VIA to Labelme conversion
