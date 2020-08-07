#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from dataset import chair_renderer, chair_renderer2, coco_loader
from img_loader.dataset import chair_renderer, coco_loader
#from utils.utils import *
#from utils.parse_config import *

def get_train_val(cfg):

    if cfg.DATASET.NAME == 'chair_trans':
        print('chair_trans')
        train = chair_renderer.chair_randomRT(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        val = chair_renderer.chair_randomRT(cfg.DATASET, 'val')
    
    elif cfg.DATASET.NAME == 'chair_trans2':
        print('chair_trans')
        train = chair_renderer.chair_randomRT(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        val = chair_renderer.chair_randomRT(cfg.DATASET, 'val')

    elif cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        train = coco_loader.coco2017(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        val = coco_loader.coco2017(cfg.DATASET, 'val')

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        train = coco_loader.coco2014(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        val = coco_loader.coco2014(cfg.DATASET, 'val')
    
    return train, val



def get_train(cfg):

    cfg.DATASET.IDS = cfg.DATASET.IDS_train

    if cfg.DATASET.NAME == 'chair_trans':
        print('chair_trans')
        train = chair_renderer.chair_randomRT(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
    
    elif cfg.DATASET.NAME == 'chair_trans2':
        print('chair_trans')
        #train = chair_renderer.chair_randomRT(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)

    elif cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        train = coco_loader.coco2017(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        train = coco_loader.coco2014(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
    
    return train


def get_val(cfg):

    cfg.DATASET.IDS = cfg.DATASET.IDS_val

    if cfg.DATASET.NAME == 'chair_trans':
        print('chair_trans')
        val = chair_renderer.chair_randomRT(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)
    
    elif cfg.DATASET.NAME == 'chair_trans2':
        print('chair_trans')
        val = chair_renderer.chair_randomRT(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)

    elif cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        val = coco_loader.coco2017(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        val = coco_loader.coco2014(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)
    

    return val



def get_test(cfg):

    if cfg.DATASET.NAME == 'chair_trans':
        test = chair_renderer.chair_randomRT(cfg.DATASET, 'test')

    elif cfg.DATASET.NAME == 'COCO2017':
        test = coco_loader.coco2017(cfg.DATASET, 'test')

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        #test = coco_loader.coco2014(DATASET, 'test')
        test = coco_loader.coco2014(cfg.DATASET, 'val')

    
    

    return test
