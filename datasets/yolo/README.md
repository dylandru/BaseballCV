# YOLO Format Available Datasets

In this directory, there are text files pointing to different zip files of different datasets, annotated in YOLO format.

---
## baseball_rubber_home_glove.txt dataset.

Contents: train, test, valid folders, each one with its respective images and labels folders.

- train: 3925 images, 3925 annotations.
- test: 88 images, 88 annotations.
- valid: 98 images, 98 annotations.

This is the dictionary of available classes: 

{0: 'glove', 1: 'homeplate', 2: 'baseball', 3: 'rubber'}



### Train Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |    3829 |
| homeplate       |    3794 |
| baseball        |    3923 |
| rubber          |    3506 |



### Test Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |      87 |
| homeplate       |      87 |
| baseball        |      87 |
| rubber          |      78 |



### Valid Dataset

| Class Name      |   Count |
|-----------------|---------|
| glove           |      95 |
| homeplate       |      91 |
| rubber          |      88 |
| baseball        |      97 |


## baseball_rubber_home.txt dataset.

Contents: train, test, valid folders, each one with its respective images and labels folders.

- train: 3905 images, 3905 annotations.
- test: 87 images, 87 annotations.
- valid: 98 images, 98 annotations.

This is the dictionary of available classes: 

{15: 'homeplate', 16: 'baseball', 17: 'rubber'}

### Train Dataset

| Class Name   |   Count |
|--------------|---------|
| homeplate    |    3794 |
| baseball     |    3923 |
| rubber       |    3506 |


### Test Dataset

| Class Name   |   Count |
|--------------|---------|
| homeplate    |      87 |
| baseball     |      87 |
| rubber       |      78 |


### Valid Dataset

| Class Name   |   Count |
|--------------|---------|
| homeplate    |      92 |
| rubber       |      89 |
| baseball     |      98 |

## baseball.txt dataset.

Contents: train, test, valid folders, each one with its respective images and labels folders.

- train: 4534 images, 4534 annotations.
- test: 426 images, 426 annotations.
- valid: 375 images, 375 annotations.

This is the dictionary of available classes: 

{0: 'glove', 1: 'homeplate', 2: 'baseball', 3: 'rubber'}

### Train Dataset

| Class Name   |   Count |
|--------------|---------|
| baseball     |    4569 |


### Test Dataset

| Class Name   |   Count |
|--------------|---------|
| baseball     |     430 |


### Valid Dataset

| Class Name   |   Count |
|--------------|---------|
| baseball     |     377 |


## OKD_NOKD.txt dataset.

These are images for catchers One Knee Down and No One Knee Down positions for classifiers.

Contents: OKD, NOKD folders.

- OKD: 1408 images.
- NOKD: 1408 images.

## test_dataset.txt dataset.

This is a test dataset for testing the model classes functionality.

Contents: Sample of images from baseball dataset.
