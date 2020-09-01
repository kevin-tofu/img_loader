# Image loader Interface
This repository is a dataloader that outputs annotations for Machine Learning purpose.  
The feature of this repository is that  
* images and its annotation that is written in COCO format can be loaded easily
* can make data augmentation if you give data augmentation function.  
  (Now, Only libraly that can be used is albumentations [ https://github.com/albumentations-team/albumentations ].)  
* can get annotations in several formats if you change a parameter.


## Requirement
* pycocotools
* albumentations
* numpy
* scikit-image

## Installation
### Install libraries to conda environment
```
conda install -c conda-forge pycocotools
conda install -c conda-forge albumentations
conda install -c conda-forge numpy
conda install -c conda-forge scikit-image
```

### Clone repository
```
git clone --recursive http://10.115.1.14/kohei/img_loader.git  
```

## Configuration of Directory Hierarchy   
Annotations should be denoted by coco format.  
```
directory (COCO****)
  |-annotations
  |
  |   |-instances_train****.json
  |   |-instances_val****.json
  |   |-person_keypoints_train****.json
  |   |-person_keypoints_val****.json
  |   |-captions_train****.json
  |   |-captions_val****.json
  |-images
  |   |-train****
  |        |-0000001.jpg
  |        |-0000002.jpg
  |        |-0000003.jpg
  |        |-.....
  |        |-.....
  |   |-val****
  |        |-0000001.jpg
  |        |-0000002.jpg
  |        |-0000003.jpg
  |        |-.....
  |        |-.....

 
```

## Usage of coco_loader.py
This class loads mini-batches that includes images with annotations from COCO format dataset.  
Data augmentation will be done by library albumentation.  
images and annotations will be transformed based on it.  
this class outputs raw data image and annotations if you won't give transformer.  


### Ex. How to load images with annotation on each mini-batch.
```
from easydict import EasyDict as edict
from dataset.augmentator import get_compose, get_compose_keypoints
from albumentations import Compose
from albumentations.augmentations.transforms import Resize

cfg = edict()
cfg.PATH = '/data/public_data/COCO2017/'
cfg.ANNTYPE = 'bbox'
cfg.BATCHSIZE = 32
h, w = 416, 416
tf = Compose([Resize(h, w, p=1.0)],\
              bbox_params={'format':'coco', 'label_fields':['category_id']})

dataloader = coco_specific(cfg, "train", tf, "2017")
imgs, annotations, dataloader.__next__()

```

### How to convert data format.

#### BBox format  
if you set form on configuration dictionary, you can change data format.
if you choose "icxywh_normalized", the annotations takes shape (i, c, x, y, w, h).  
where each numbers of elements is  
i : batch number  
c : category number  
x : left-upper x-normalized-coordinate of bbox  
y : left-upper y-normalized-coordinate of bbox  
w : normalized width of bbox  
h : normalized height of bbox  
```
cfg.ANNTYPE = 'bbox'
cfg.FORM = "icxywh_normalized"
dataloader = coco_specific(cfg, "train", tf, "2017")
imgs, annotations, dataloader.__next__()
```

if you choose "x1y1whc", the annotations will be returned as a list,  
and its each elements are correspoinding to the each batch of images, and it takes the list of "(x1, y1, w, h, c)".  
where each numbers of elements is  
x1 : left-upper x-coordinate of bbox  
y1 : left-upper y-coordinate of bbox  
w : width of bbox  
h : height of bbox  
c : category number  
```
cfg.FORM = "x1y1whc"

# or change form variable.
#dataloader.form = "x1y1whc"
```


#### Keypoints format  
you will get keypoints annotations if you set cfg.ANNTYPE = 'keypoints'.
```
cfg.ANNTYPE = 'keypoints'
cfg.FORM = "xyc"

dataloader = coco_specific(cfg, "train", tf, "2017")
imgs, annotations, dataloader.__next__()
```

### How to test coco_loader.py code

#### coco_loader.py
Default dataset path is /data/public_data/COCO2017/

* check loading  
python img_loader/dataset/coco_loader.py loader path2COCO

* check keypoints  
python img_loader/dataset/coco_loader.py keypoints path2COCO

* check bbox  
python img_loader/dataset/coco_loader.py bbox path2COCO
