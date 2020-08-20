# Image loader Interface
 
## Usage of coco_loader.py


This class loads mini-batches that includes images with annotations from COCO format dataset.  
Data augmentation will be done by library albumentation.  
images and annotations will be transformed based on it.  
this class outputs raw data image and annotations if you won't give transformer.  

### Requirement
* pycocotools
* numpy
* scikit-image
* albumentation

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
              bbox_params={'format':format, 'label_fields':['category_id']})

dataloader = coco_base(cfg, "train", tf, "2017")
imgs, annotations, dataloader.__next__()

```

### How to convert data format.

if you change the valibalbe .format, you can change data format.
```
dataloader = coco_base(cfg, "train", tf, "2017")
dataloader.form = "icxywh_normalized"
#dataloader.form = "x1y1whc"
imgs, annotations, dataloader.__next__()
```
if you choose "icxywh_normalized", the annotations takes shape (i, c, x, y, w, h).  
i : batch number  
c : category number  
x : left-upper x-coordinate of bbox  
y : left-upper y-coordinate of bbox  
w : width of bbox  
h : height of bbox  


### How to Test coco_loader.py code

#### coco_loader.py
Default dataset path is /data/public_data/COCO2017/

* check loading  
python img_loader/dataset/coco_loader.py loader path2COCO

* chekk keypoints  
python img_loader/dataset/coco_loader.py keypoints path2COCO

* chekk bbox  
python img_loader/dataset/coco_loader.py bbox path2COCO
