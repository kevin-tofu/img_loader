import albumentations as A 
from albumentations import Compose
from albumentations.augmentations.transforms import Resize, HorizontalFlip, RandomSizedCrop, HueSaturationValue
from albumentations.augmentations.transforms import RandomGamma, Blur, RGBShift, GaussNoise, ChannelShuffle
from albumentations.augmentations.transforms import Normalize, ShiftScaleRotate,ChannelShuffle, RandomResizedCrop

# Resize image to (image_height, image_width) with 100% probability
# Flip LR with 50% probability
# Crop image and resize image to (image_height, image_width) with 100% probability
# Change HSV from -hue_shift to +hue_shift and so on with 100% probability
# Format 'pascal_voc' means label is given like [x_min, y_min, x_max, y_max]
# format should be 'coco', 'pascal_voc', or 'yolo'


def get_compose_resize0(image_height, image_width, format):
    return Compose([Resize(image_height, image_width, p=1.0)],\
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose_resize(image_height, image_width, format):
    return Compose([Resize(image_height, image_width, p=1.0), \
                    Normalize(always_apply=True)],\
                    bbox_params={'format':format, 'label_fields':['category_id']})


def get_compose_resize2(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):

    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=0.2),\
                    Blur(p=0.05), \
                    GaussNoise(p=0.20), \
                    RandomGamma(p=0.1), \
                    RGBShift(p=0.1), \
                    ChannelShuffle(p=0.1),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5), \
                    Normalize(always_apply=True)], \
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose_resize3(crop_min_max, image_height, image_width, format):

    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=0.2),\
                    Blur(p=0.05), \
                    GaussNoise(p=0.25), \
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), \
                    Normalize(always_apply=True)], \
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose_resize4(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
    _c_r = (-5, 5)
    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=0.7),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.9), \
                    Normalize(always_apply=True)\
                    ], \
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose_resize5(image_height, image_width, format):
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
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.9), \
                    RandomResizedCrop(resize_to2[0], resize_to2[1], scale=crop_scale, ratio=crop_ratio, always_apply=True),\
                    ], \
                    bbox_params={'format':fmt, 'label_fields':['category_id']})


def get_compose_keypoints0(image_height, image_width):
    _c_r = (-10, 10)
    crop_scale = (0.5, 1.0)
    crop_ratio = (0.75, 1.333)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10
    resize_to2 = (image_height, image_width)
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
                    keypoint_params=A.KeypointParams(format='xy'))


def get_compose(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):

    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=0.1),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    RandomGamma(p=0.1), \
                    GaussNoise(p=0.2), \
                    Blur(p=0.1), \
                    RGBShift(p=0.1), \
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5), \
                    Normalize(always_apply=True)], \
                    bbox_params={'format':format, 'label_fields':['category_id']})

#def get_compose_bbox(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
def get_compose0(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=1.0),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=1.0), \
                    Normalize(always_apply=True)],\
                    bbox_params={'format':format, 'label_fields':['category_id']}, 
                    keypoint_params=A.KeypointParams(format='xy'))


def get_compose_keypoints(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):

    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, w2h_ratio=0.75, p=0.2),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.2),\
                    RandomGamma(p=0.5), \
                    GaussNoise(p=0.2), \
                    Blur(p=0.3), \
                    RGBShift(p=0.9), \
                    Normalize(always_apply=True)], \
                    keypoint_params=A.KeypointParams(format='xy'))



def get_compose_keypoints4(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
    _c_r = (-5, 5)
    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    #RandomSizedCrop(crop_min_max, image_height, image_width, p=0.2),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, w2h_ratio=0.75, p=0.7),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=40, p=0.9), \
                    Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))


def get_compose_keypoints5(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
    _c_r = (-8, 8)
    return Compose([HorizontalFlip(p=0.5),\
                    Blur(p=0.2), \
                    GaussNoise(p=0.7), \
                    RandomGamma(gamma_limit = (95, 105) , p=0.6), \
                    RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.6), \
                    ChannelShuffle(p=0.4),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.1),\
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=40, p=0.9), \
                    Resize(image_height, image_width, p=1.0),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, w2h_ratio=0.75, p=0.7),\
                    Normalize(always_apply=True)\
                    ], \
                    keypoint_params=A.KeypointParams(format='xy'))

def get_compose_keypoints_resize(image_height, image_width, format):
    return Compose([Resize(image_height, image_width, p=1.0), \
                    Normalize(always_apply=True)],\
                    keypoint_params=A.KeypointParams(format='xy'))

