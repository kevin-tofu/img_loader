#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from dataset import chair_renderer, chair_renderer2, coco_loader
from img_loader.dataset import coco_loader
from torch.utils.data import DataLoader
import random


def get_train(cfg, transformer):

    cfg.DATASET.IDS = cfg.DATASET.IDS_train

    if cfg.DATASET.NAME == 'COCO':
        print('COCO 2017 ')
        data_ = coco_loader.coco_base_specific_(cfg.DATASET, 'train', transformer)
        train = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                            shuffle=True, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                            worker_init_fn=lambda x: random.seed())

    else:
        raise ValueError(cfg.DATASET.NAME, "img_loader/load_coco.py" + " you should choose name from 'COCO2017', 'COCO2014', 'COCO_original'")

    return train


def get_val(cfg, transformer):

    cfg.DATASET.IDS = cfg.DATASET.IDS_val

    if cfg.DATASET.NAME == 'COCO':
        print('COCO 2017 ')
        data_ = coco_loader.coco_base_specific_(cfg.DATASET, 'val', transformer)
        val = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                         shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                         worker_init_fn=lambda x: random.seed())

    else:
        raise ValueError(cfg.DATASET.NAME, "img_loader/load_coco.py" + " you should choose name from 'COCO2017', 'COCO2014', 'original'")

    return val



def get_test(cfg, transformer):

    cfg.DATASET.IDS = cfg.DATASET.IDS_test

    if cfg.DATASET.NAME == 'COCO':
        print('COCO 2017 ')
        data_ = coco_loader.coco2017_(cfg.DATASET, 'test', transformer)
        loader = DataLoader(data_, batch_size=cfg.DATASET.BATCHSIZE,\
                            shuffle=False, num_workers=cfg.DATASET.WORKERS, collate_fn=data_.collate_fn,
                            worker_init_fn=lambda x: random.seed())

    else:
        raise ValueError(cfg.DATASET.NAME, "")



    return loader

