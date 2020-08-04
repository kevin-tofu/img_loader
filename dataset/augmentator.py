
from albumentations import Compose
from albumentations.augmentations.transforms import Resize, HorizontalFlip, RandomSizedCrop, HueSaturationValue
from albumentations.augmentations.transforms import RandomGamma, Blur, RGBShift, GaussNoise, ChannelShuffle

# Resize image to (image_height, image_width) with 100% probability
# Flip LR with 50% probability
# Crop image and resize image to (image_height, image_width) with 100% probability
# Change HSV from -hue_shift to +hue_shift and so on with 100% probability
# Format 'pascal_voc' means label is given like [x_min, y_min, x_max, y_max]
# format should be 'coco', 'pascal_voc', or 'yolo'

def get_compose_resize(image_height, image_width, format):
    return Compose([Resize(image_height, image_width, p=1.0)],\
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose0(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):
    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=1.0),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=1.0)],\
                    bbox_params={'format':format, 'label_fields':['category_id']})

def get_compose(crop_min_max, image_height, image_width, hue_shift, saturation_shift, value_shift, format):

    return Compose([Resize(image_height, image_width, p=1.0),\
                    HorizontalFlip(p=0.5),\
                    RandomSizedCrop(crop_min_max, image_height, image_width, p=0.2),\
                    HueSaturationValue(hue_shift, saturation_shift, value_shift, p=0.2),\
                    RandomGamma(p=0.5), \
                    GaussNoise(p=0.2), \
                    Blur(p=0.3), \
                    RGBShift(p=0.2)], \
                    bbox_params={'format':format, 'label_fields':['category_id']})

