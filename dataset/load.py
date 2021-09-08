#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from dataset import chair_renderer, chair_renderer2, coco_loader
from img_loader.dataset import chair_renderer, coco_loader, cricket_loader
from torch.utils.data import DataLoader
#from utils.utils import *
#from utils.parse_config import *
import random


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

    if cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        data_ = coco_loader.coco2017_(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        train = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                            shuffle=True, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                            worker_init_fn=lambda x: random.seed())

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        data_ = coco_loader.coco2014_(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        train = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=True, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())
    
    elif cfg.DATASET.NAME == 'cocoCricket' or cfg.DATASET.NAME == 'cocoCricket_sep':
        print('cocoCricket')
        data_ = cricket_loader.cocoCricket(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR)
        train = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                            shuffle=True, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                            worker_init_fn=lambda x: random.seed())
    else:
        val = None

    return train


def get_val(cfg):

    cfg.DATASET.IDS = cfg.DATASET.IDS_val

    if cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        data_ = coco_loader.coco2017_(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)
        val = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())

    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        data_ = coco_loader.coco2014_(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val)
        val = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())

    elif cfg.DATASET.NAME == 'cocoCricket':
        print('cocoCricket')
        data_ = cricket_loader.cocoCricket(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR_val)
        #data_ = cricket_loader.cocoCricket(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR)
        val = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())
    else:
        #return NotImplementedError()
        val = None

    return val



def get_test(cfg):

    #dtype = "test"
    dtype = "val"
    cfg.DATASET.IDS = cfg.DATASET.IDS_test

    if cfg.DATASET.NAME == 'COCO2017':
        print('COCO 2017 ')
        data_ = coco_loader.coco2017_(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val, cfg.DATASET.CROP)
        loader = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                            shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                            worker_init_fn=lambda x: random.seed())


    elif cfg.DATASET.NAME == 'COCO2014':
        print('COCO 2014 ')
        data_ = coco_loader.coco2014_(cfg.DATASET, 'val', cfg.DATASET.AUGMENTATOR_val, cfg.DATASET.CROP)
        loader = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())

    elif cfg.DATASET.NAME == 'cocoCricket':
        print('cocoCricket')
        data_ = cricket_loader.cocoCricket(cfg.DATASET, 'train', cfg.DATASET.AUGMENTATOR_test)
        loader = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())

    else:
        #return NotImplementedError()
        loader = None


    return loader

