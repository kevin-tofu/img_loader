
import albumentations as A 
from albumentations import Compose
from albumentations.augmentations.transforms import Resize, HorizontalFlip, RandomSizedCrop, HueSaturationValue
from albumentations.augmentations.transforms import RandomGamma, Blur, RGBShift, GaussNoise, ChannelShuffle
from albumentations.augmentations.transforms import Normalize, ShiftScaleRotate,ChannelShuffle, RandomResizedCrop
import random
import numpy as np

def aug_yolov3(resize=416):

    def temp(image_height, image_width):
        _c_r = (-10, 10)
        # Crop 60 - 100% of image
        #crop_scale = (0.3, 1.0)
        crop_scale = (0.5, 1.0)
        crop_ratio = (0.75, 1.333)
        # HSV shift limits
        hue_shift = 10
        saturation_shift = 10
        value_shift = 10
        fmt = "coco"
        _c_r = (-10, 10)
        #resize_to1 = (256*2, 192*2)
        resize_to2 = (resize, resize)

        # HSV shift limits
        hue_shift = 10
        saturation_shift = 10
        value_shift = 10
        cst_shift = 8

        #print(image_height, image_width)#, w2h_ratio=0.75
        return Compose([HorizontalFlip(p=0.5),\
                        Blur(p=0.2), \
                        GaussNoise(p=0.7), \
                        RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                        RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                        ChannelShuffle(p=0.4),\
                        HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=35, p=0.5), \
                        RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, always_apply=True),\
                        Normalize(always_apply=True)\
                        ], \
                        bbox_params={'format':fmt, 'label_fields':['category_id']})
    return temp
                        
def compose_bbox1(image_height, image_width):
    _c_r = (-10, 10)
    # Crop 60 - 100% of image
    #crop_scale = (0.3, 1.0)
    crop_scale = (0.7, 1.0)
    crop_ratio = (0.75, 1.333)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    fmt = "coco"
    _c_r = (-10, 10)
    #resize_to1 = (256*2, 192*2)
    resize_to2 = (416, 416)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, always_apply=True),\
                    Normalize(always_apply=True)\
                    ], \
                    bbox_params={'format':fmt, 'label_fields':['category_id']})


def compose_bbox2(image_height, image_width):
    _c_r = (-10, 10)
    # Crop 60 - 100% of image
    #crop_scale = (0.3, 1.0)
    crop_scale = (0.5, 1.0)
    crop_ratio = (0.75, 1.333)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    fmt = "coco"
    _c_r = (-10, 10)
    #resize_to1 = (256*2, 192*2)
    resize_to2 = (416, 416)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=35, p=0.5), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, always_apply=True),\
                    Normalize(always_apply=True)\
                    ], \
                    bbox_params={'format':fmt, 'label_fields':['category_id']})

def compose_bbox3(image_height, image_width):

    _c_r = (-10, 10)
    # Crop 60 - 100% of image
    #crop_scale = (0.3, 1.0)
    crop_scale = (0.7, 1.0)
    crop_ratio = (0.75, 1.333)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    fmt = "coco"
    _c_r = (-10, 10)
    #resize_to1 = (256*2, 192*2)
    resize_to2 = (416, 416)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, always_apply=True),\
                    ], \
                    bbox_params={'format':fmt, 'label_fields':['category_id']})

def compose_bbox_test(image_height, image_width):
    resize_to = (416, 416)
    fmt = "coco"
    return Compose([Resize(resize_to[0], resize_to[1], always_apply=True), \
                    Normalize(always_apply=True)],\
                    bbox_params={'format':fmt, 'label_fields':['category_id']})


def compose_bbox_test2(image_height, image_width):
    resize_to = (416, 416)
    fmt = "coco"
    return Compose([Resize(resize_to[0], resize_to[1], always_apply=True)], \
                    bbox_params={'format':fmt, 'label_fields':['category_id']})



def compose_keypoints1(image_height, image_width):

    _c_r = (-10, 10)
    resize_to1 = (256*2, 192*2)
    resize_to2 = (256, 192)

    image_height2 = resize_to2[0]
    # Crop 60 - 100% of image
    crop_min = image_height2*30//100
    crop_max = image_height2*2
    crop_min_max = (crop_min, crop_max)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.9), \
                    Resize(resize_to1[0], resize_to1[1], p=1.0),\
                    RandomSizedCrop(crop_min_max, resize_to2[0], resize_to2[1], w2h_ratio=0.75, always_apply=True),\
                    Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))



def compose_keypoints2(image_height, image_width):

    _c_r = (-10, 10)
    #resize_to1 = (256*2, 192*2)
    resize_to2 = (256, 192)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=(0.4, 1.0), always_apply=True),\
                    Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))


def compose_keypoints3(image_height, image_width):

    _c_r = (-10, 10)
    resize_to2 = (256, 192)

    crop_scale = (0.85, 1.0)
    crop_ratio = (0.9, 1.1)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, p=0.2),\
                    Resize(resize_to2[0], resize_to2[1], always_apply=True),\
                    Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))


def compose_keypoints3_test(image_height, image_width):

    _c_r = (-10, 10)
    resize_to2 = (256, 192)

    crop_scale = (0.85, 1.0)
    crop_ratio = (0.9, 1.1)

    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    cst_shift = 8

    #print(image_height, image_width)#, w2h_ratio=0.75
    return Compose([Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=cst_shift, g_shift_limit=cst_shift, b_shift_limit=cst_shift, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio = crop_ratio, p=0.2),\
                    Resize(resize_to2[0], resize_to2[1], always_apply=True),\
                    #Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))

def compose_keypoints_test(image_height, image_width):
    resize_to = (256, 192)
    return Compose([Resize(resize_to[0], resize_to[1], p=1.0), \
                    Normalize(always_apply=True)],\
                    keypoint_params=A.KeypointParams(format='xy'))



def compose_keypoints_resize(image_height, image_width):
    resize_to = (256, 192)
    return Compose([Resize(resize_to[0], resize_to[1], p=1.0)],\
                    keypoint_params=A.KeypointParams(format='xy'))

class aug_bbox(object):
    def __init__(self, compose_func):
        self.compose_func = compose_func

    def __call__(self, image, bboxes, category_id):
        
        compose = self.compose_func(image.shape[0], image.shape[1])
        img_augmented = compose(image=image, bboxes=bboxes, category_id=category_id)

        return img_augmented

class bbox_test(object):
    def __init__(self):
        pass
    def __call__(self, image, bboxes, category_id):

        compose = compose_bbox_test(image.shape[0], image.shape[1])
        #compose = compose_bbox_test(len(image), len(image[0]))
        img_augmented = compose(image=image, bboxes=bboxes, category_id=category_id)

        return img_augmented


class aug_keypoints(object):
    def __init__(self, compose_func):
        self.compose_func = compose_func

    def __call__(self, image, keypoints):

        compose = self.compose_func(image.shape[0], image.shape[1])
        img_augmented = compose(image=image, keypoints=keypoints)

        return img_augmented


#"keypoints": [
#                "nose",
#                "left_eye",
#                "right_eye",
#                "left_ear",
#                "right_ear",
#                "left_shoulder",
#                "right_shoulder",
#                "left_elbow",
#                "right_elbow",
#                "left_wrist",
#                "right_wrist",
#                "left_hip",
#                "right_hip",
#                "left_knee",
#                "right_knee",
#                "left_ankle",
#                "right_ankle"
#            ]

class aug_keypoints_flipped(object):
    def __init__(self, compose_func, p = 0.5):
        self.compose_func = compose_func
        self.__right = [2, 4, 6, 8, 10, 12, 14, 16]
        self.__left = [1, 3, 5, 7, 9, 11, 13, 15]
        self.__p = 1.0 - p

    def __call__(self, image, keypoints):

        compose = self.compose_func(image.shape[0], image.shape[1])
        img_augmented = compose(image=image, keypoints=keypoints)
        if random.random() > self.__p:
        #if random.random() > 0.0:
            key_flipped = []
            x_size = img_augmented["image"].shape[1] - 1
            for k in img_augmented["keypoints"]:
                x, y = k[0], k[1]
                x_new = x_size - x
                c = k[2]
                if k[3] in self.__right:
                    j = k[3] - 1
                elif k[3] in self.__left:
                    j = k[3] + 1
                else:
                    j = k[3]
                
                key_flipped.append([x_new, y, c, j])

            img_augmented["image"] = img_augmented["image"][:, ::-1]
            img_augmented["keypoints"] = key_flipped

        return img_augmented


class keypoints_test(object):
    def __init__(self):
        pass
    def __call__(self, image, keypoints):

        compose = compose_keypoints_test(image.shape[0], image.shape[1])
        img_augmented = compose(image=image, keypoints=keypoints)

        return img_augmented

def draw_box(img, target, _print=True):
    
    from skimage.draw import rectangle, rectangle_perimeter
    ret = np.copy(img)
    if len(target) == 0:
        return ret

    for t in target:
        #print(t)
        w = int(t[2])
        h = int(t[3])
        x1 = int(t[0] - w / 2)
        y1 = int(t[1] - h / 2)
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        if _print:
            print(x1, y1, x2, y2)

        x1 = max(x1, 5)
        y1 = max(y1, 5)
        x2 = min(x2, ret.shape[1] - 5)
        y2 = min(y2, ret.shape[0] - 5)

        color_line = np.array([255, 0, 0], dtype=np.uint8)
        rr, cc = rectangle_perimeter(start = (y1, x1), end = (y2, x2))
         
        ret[rr, cc] = color_line
    return ret

def draw_keypoints(img, ann, _print=True):
    from skimage.draw import circle

    __right = [2, 4, 6, 8, 10, 12, 14, 16]
    __left = [1, 3, 5, 7, 9, 11, 13, 15]
    ret = np.copy(img)
    for a in ann:
        for xy in a:
            x = int(xy[0])
            y = int(xy[1])
            c = int(xy[2])
            j = int(xy[3])
            rr, cc = circle(y, x, 5, ret.shape)
            if _print:
                print(xy)
            
            if j in __right:
                if xy[2] == 1:
                    _color = (0, 64, 0)
                elif xy[2] == 2:
                    _color = (0, 255, 0)
            elif j in __left:
                if xy[2] == 1:
                    _color = (64, 0, 0)
                elif xy[2] == 2:
                    _color = (255, 0, 0)
            else:
                _color = (0, 0, 255)

            ret[rr, cc, :] = _color
    return ret

def check_bbox(path, cfg, dataset):

    from torch.utils.data import DataLoader
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    random.seed(0)
    
    #compose = aug_bbox(compose_bbox2)
    compose = aug_bbox(compose_bbox3)
    #compose = aug_bbox(compose_bbox_test)
    #compose = aug_bbox(compose_bbox_test2)
    #compose = None
    
    data_type = 'val'
    data_aug = dataset(cfg, data_type, compose)
    data_aug.initialize_loader()
    data_normal = dataset(cfg, data_type, None)
    data_normal.initialize_loader()

    loader_aug = DataLoader(data_aug, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_aug.collate_fn)
    loader_normal = DataLoader(data_normal, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_normal.collate_fn)

    pl.figure()
    pl.figure(figsize=(12, 5), dpi=300)

    for i, ((img_aug, target_aug), (img, target)) in enumerate(zip(loader_aug, loader_normal)):
        print(len(img))
        #__box = target[0]
        if i > 10:
            break
        for ii, (c_aug, t_aug, c_normal, t_normal) in enumerate(zip(img_aug, target_aug[0], img, target[0])):
            
            fname = path + str(i*32 + ii) + ".jpg"
            print(fname)
            c_box_aug = draw_box(c_aug, t_aug, True)
            c_box_normal = draw_box(c_normal, t_normal, False)

            pl.clf()
            pl.subplot(131)
            pl.imshow(c_normal)

            pl.subplot(132)
            pl.imshow(c_box_normal)
            
            pl.subplot(133)
            pl.imshow(c_box_aug)
            pl.savefig(fname)


def check_keypoints(path, cfg, dataset):

    
    from torch.utils.data import DataLoader
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    num_seed = 0
    random.seed(num_seed)
    
    #compose = aug_bbox(compose_bbox2)
    #compose = aug_keypoints(compose_keypoints_test)
    #compose = aug_keypoints(compose_keypoints2)
    #compose = aug_keypoints(compose_keypoints3)
    #compose = aug_keypoints_flipped(compose_keypoints3)
    compose = aug_keypoints_flipped(compose_keypoints3_test, p = 1.0)
    
    
    #compose = aug_keypoints(compose_keypoints_resize)

    data_type = 'val'
    data_aug = dataset(cfg, data_type, compose)
    data_aug.initialize_loader()
    data_normal = dataset(cfg, data_type, None)
    data_normal.initialize_loader()

    loader_aug = DataLoader(data_aug, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_aug.collate_fn,
                        worker_init_fn=lambda x: random.seed(num_seed))
    loader_normal = DataLoader(data_normal, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_normal.collate_fn,
                        worker_init_fn=lambda x: random.seed(num_seed))

    pl.figure()
    pl.figure(figsize=(12, 5), dpi=300)

    for i, ((img_aug, target_aug), (img, target)) in enumerate(zip(loader_aug, loader_normal)):
        print(len(img))
        #__box = target[0]
        targets_aug = target_aug[0]
        targets_nor, ids_nor = target[0], target[1]

        if i > 10:
            break
        for ii, (c_aug, t_aug, c_normal, t_normal, id_aug) in enumerate(zip(img_aug, targets_aug, img, targets_nor, ids_nor)):
            
            fname = path + str(i*32 + ii) + ".jpg"
            print(fname)
            c_key_aug = draw_keypoints(c_aug, t_aug, True)
            c_key_normal = draw_keypoints(c_normal, t_normal, False)

            img_original = data_normal.get_img(id_aug)

            pl.clf()
            pl.subplot(131)
            pl.imshow(img_original)

            pl.subplot(132)
            pl.imshow(c_key_normal)
            
            pl.subplot(133)
            pl.imshow(c_key_aug)
            pl.savefig(fname)


def main_bbox():

    from easydict import EasyDict as edict
    import sys, os, shutil
    #sys.path.append('../img_loader/dataset/')
    sys.path.append('./img_loader/dataset/')
    import coco_loader

    path = "./dataset/temp_bbox/"
    if os.path.exists(path) == True:
        shutil.rmtree(path)
    if os.path.exists(path) == False:
        os.makedirs(path)
    year = '2017'
    cfg = edict()
    cfg.PATH = '/data/public_data/COCO2017/'
    cfg.ANNTYPE = 'bbox'
    cfg.IDS = 'all'
    cfg.IDS = 'vehicle'
    cfg.IDS = 'vehicle_all'
    cfg.BATCHSIZE = 30
    cfg.NUM_CLASSES = 80
    coco = coco_loader.coco2017_
    check_bbox(path, cfg, coco)


def main_keypoints():
    from easydict import EasyDict as edict
    import sys, os, shutil
    #sys.path.append('../img_loader/dataset/')
    sys.path.append('./img_loader/dataset/')
    import coco_loader

    path = "./dataset/temp_keypoints/"
    if os.path.exists(path) == True:
        shutil.rmtree(path)
    if os.path.exists(path) == False:
        os.makedirs(path)
    year = '2017'
    cfg = edict()
    cfg.PATH = '/data/public_data/COCO2017/'
    cfg.ANNTYPE = 'keypoints'
    cfg.IDS = 'keypoints'
    cfg.BATCHSIZE = 30
    cfg.NUM_CLASSES = 80
    coco = coco_loader.coco2017_
    check_keypoints(path, cfg, coco)
    

if __name__ == '__main__':

    #main_bbox()
    main_keypoints()


    