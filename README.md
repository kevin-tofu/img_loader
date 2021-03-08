
# Image loader Interface
This repository is a dataloader that outputs annotations for Machine Learning purpose.  
The feature of this repository is that  
* export images and its annotation that is written in COCO format. It can be loaded easily for deep learning process
* support bbox and keypoint annotation.
* can make data augmentation if you give data augmentation function to loader class.  
  Now, external libraly (like albumentations) can be applied. 
   - https://github.com/albumentations-team/albumentations
* can get annotations in several formats if you change parameters.


## Requirement
* pycocotools
* numpy
* scikit-image

### Sub-tools
* albumentations


## Installation
### Install libraries to conda environment
```
conda install -c conda-forge pycocotools
conda install -c conda-forge numpy
conda install -c conda-forge scikit-image
```

### Install sub-tools
```
conda install -c conda-forge albumentations
```

### Clone repository
```
git clone --recursive http://10.115.1.14/kohei/img_loader.git  
```

## Directory Hierarchy Configuration   
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

## Collate functions for aggregating information.
It is heavy task to load inforamtion such as many images or vidoes.
So, Some dataloaders like tensorflow or Torch have funcation which is going to load information by multi-processing on CPU.
Users indicates how many works to use for loading, and avoid this part becomes bottle-neck on processes.
For using this function, users need to give a collate function to dataloader.
The collate function is going to aggregate information that is loaded by each workers on multi-processing.
 This repository gives you some collate function choices.


|Collate functions|Explanations|
|:---:|:---|
|collate_fn_bbox| returns [images, (bbox, img_id, imsize)].|
|collate_fn_keypoints| returns [images, (keypoints, img_id, center, scale)] <br> center means center coordinate of bbox, and scale means bbox scale.|
|collate_fn_images| returns [(None, img_id, imsize)] <br>it will be used for dataset without annotation.|
|collate_fn_images_sub| returns [(None, img_id, imsize, img_fname)]|


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

### Ex. How to load images using Dataloader in pytorch.
```
from dataset import data_loader
from torch.utils.data import DataLoader, Dataset
data_ = coco_specific(cfg, 'train', tf, "2017")
data_.initialize_loader()
loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                    shuffle=False, num_workers=5, collate_fn = data_loader.collate_fn_bbox)

for batch_idx, (imgs, targets_list) in enumerate(loader):
      print(np.array(imgs).shape)
      for targets in targets_list[0]:
          print(np.array(targets).shape)

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

### How to load specific images
Custom loader should be inherited If you want to load specific images in datasets


### How to test coco_loader.py code

#### coco_loader.py
Default dataset path is /data/public_data/COCO2017/

* check loading  
python img_loader/dataset/coco_loader.py loader path2COCO

* check keypoints  
python img_loader/dataset/coco_loader.py keypoints path2COCO

* check bbox  
python img_loader/dataset/coco_loader.py bbox path2COCO


### How to use data augmentator.
There are 3 augmentator on this repository.

This is for albumentation library.
  aug_bbox, aug_keypoints, aug_keypoints_flipped

aug_keypoints_flipped is going to make flip keypoints based on human keypoint.
If you flip an image with keypoint, right keypoint should move to left keypoint. 

For example, left elbow -> right elbow. right elbow -> left knee.
But, albumentation will not do this sort of opearations.
So, aug_keypoints_flipped will do this operation instead of albumentation.


#### Interface for albumentation
```
from albumentations import Compose
from albumentations.augmentations.transforms import HorizontalFlip, Blur, Normalize

def f_augmentation1(image_height, image_width):
  return Compose([HorizontalFlip(p=0.5), Blur(p=0.2), Normalize(always_apply=True), \
                  bbox_params={'format':fmt, 'label_fields':['category_id']})

def f_augmentation2(image_height, image_width):
    return Compose([HorizontalFlip(p=0.5), Blur(p=0.2)], \
                    keypoint_params=A.KeypointParams(format='xy'))

augmentator_bbox = aug_bbox(f_augmentation1)
augmentator_kpoint1 = aug_keypoints(f_augmentation2)
augmentator_kpoint2 = aug_keypoints_flipped(f_augmentation2, p = 1.0)


```



