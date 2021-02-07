"""
Inspect VIA .json annotations 
"""
# Import libraries
import json

# Load json annotations
path_to_annotations = 'via_annotations.json'

with open(path_to_annotations, 'r') as f:
    annotations = f.read()
annotations = json.loads(annotations)

# Perform operations with annotations
# annotations = list(annotations.values()) # Transform to the list
