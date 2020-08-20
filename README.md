# Usage
## coco_loader.py

load a batch that includes images with annotations.  
Data augmentation will be done by library albumentation.  
images and annotations will be transformed based on it.  
this class outputs raw data image and annotations if you won't give transformer.  

#### Requirement
* pycocotools
* numpy
* scikit-image


#### Ex. How to load images with annotation on each mini-batch.
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



## How to Test code

#### coco_loader.py


* check loading  
python img_loader/dataset/coco_loader.py loader path2COCO

* chekk keypoints  
python img_loader/dataset/coco_loader.py keypoints path2COCO

* chekk bbox  
python img_loader/dataset/coco_loader.py bbox path2COCO
