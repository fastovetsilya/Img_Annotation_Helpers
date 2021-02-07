"""
Inspect VIA .json annotations 
"""
# Import libraries
import json

# Load json annotations
path_to_annotations = 'tomato.json'

with open(path_to_annotations, 'r') as f:
    annotations = f.read()
annotations = json.loads(annotations)


