# COCO Format Available Datasets

In this directory, there are text files pointing to different zip files of different datasets, annotated in COCO format.

---
## baseball_rubber_home_glove_COCO.txt dataset.

Contents: train, test, val folders, each one with its respective images and labels folders.

- train: 3475 images, 3475 annotations.
- test: 408 images, 408 annotations.
- val: 206 images, 206 annotations.

There is a COCO_annotations folder for the three instances annotation files.

This is the category distribution in the annotations files for the available classes: 

    "categories": [
        {
            "id": 0,
            "name": "glove",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "homeplate",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "baseball",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "rubber",
            "supercategory": "none"
        }
    ]

### Train Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |    3406 |
| homeplate       |    3375 |
| baseball        |    3493 |
| rubber          |    3121 |

### Test Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |     402 |
| homeplate       |     395 |
| baseball        |     408 |
| rubber          |     372 |



### Val Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |     203 |
| homeplate       |     202 |
| baseball        |     206 |
| rubber          |     179 |


---
## baseball_rubber_home_COCO.txt dataset.

Contents: train, test, val folders, each one with its respective images and labels folders.

- train: 3994 images, 3994 annotations.
- test: 774 images, 774 annotations.
- val: 400 images, 400 annotations.

There is a COCO_annotations folder for the three instances annotation files.

This is the category distribution in the annotations files for the available classes: 

    "categories": [
        {
            "id": 0,
            "name": "glove",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "homeplate",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "baseball",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "rubber",
            "supercategory": "none"
        }
    ]

### Train Dataset

| Class Name   |   Count |
|--------------|---------|
| homeplate    |    3879 |
| baseball     |    4012 |
| rubber       |    3582 |


### Test Dataset

| Class Name   |   Count |
|--------------|---------|
| homeplate    |     750 |
| baseball     |     777 |
| rubber       |     695 |


### Val Dataset

| Class Name   |   Count |
|--------------|---------|
| baseball     |     400 |
| homeplate    |     390 |
| rubber       |     367 |

---
## base_detector_coco.txt dataset.

Contents: `images/` (660 images) and `annotations/` with a single `instances_default.json` annotation file. No train/test/val split — a single flat set.

- images: 660
- annotations: 668

This is the category distribution in the annotations file for the available classes: 

    "categories": [
        {
            "id": 0,
            "name": "base",
            "supercategory": ""
        }
    ]

### Annotations

| Class Name   |   Count |
|--------------|---------|
| base         |    668  |

Note: the category id was normalized from 1 to 0 to match the repository's COCO convention.
