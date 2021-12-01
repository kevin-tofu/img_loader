
import sys
import os
import random

import numpy as np
from pycocotools.coco import COCO
from skimage import io
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import data_loader
import time
from torch.utils.data import DataLoader, Dataset
import albumentations as A 
from albumentations import Compose
from albumentations.augmentations.crops.transforms import Crop


if __name__ == '__main__':
    from coco_selection import* 
else:
    from img_loader.dataset.coco_selection import* 

def func_coco2mpii(joints, __map):

    #(N, xycn)
    joints_new = [[j[0], j[1], j[2], __map[j[3]]] for j in joints]
    joints_new = sorted(joints_new, reverse=False, key=lambda x: x[3]) 
    return joints_new


def draw_box(img, target, fmt="xywhc"):

    #from skimage.draw import line
    from skimage.draw import rectangle, rectangle_perimeter
    
    #rr, cc = [], []
    ret = np.copy(img)
    if len(target) == 0:
        return ret

    for t in target:
        if fmt == "x1y1whc":
            x1 = int(t[0])
            y1 = int(t[1])
            x2 = int(t[2]) + x1
            y2 = int(t[3]) + y1

        elif fmt == "xywhc":
            w = int(t[2])
            h = int(t[3])
            x1 = int(t[0] - w / 2)
            y1 = int(t[1] - h / 2)
            x2 = int(t[0] + w / 2)
            y2 = int(t[1] + h / 2)

        elif fmt == "xywhc_normalized":
            w = int(t[2] * (img.shape[1] - 1))
            h = int(t[3] * (img.shape[0] - 1))
            x1 = int(t[0] * (img.shape[1] - 1) - w / 2)
            y1 = int(t[1] * (img.shape[0] - 1) - h / 2)
            x2 = int(t[0] * (img.shape[1] - 1) + w / 2) - 1
            y2 = int(t[1] * (img.shape[0] - 1) + h / 2) - 1

        x1 = np.clip(x1, 0, ret.shape[1] - 4)
        x2 = np.clip(x2, 0, ret.shape[1] - 4)
        y1 = np.clip(y1, 0, ret.shape[0] - 4)
        y2 = np.clip(y2, 0, ret.shape[0] - 4)
        print(x1, y1, x2, y2, ret.dtype, ret.shape)
        color_line = np.array([255, 0, 0], dtype=np.uint8)

        #rr, cc = rectangle(start = (y1, x1), end = (y2 - 1, x2 - 1))
        rr, cc = rectangle_perimeter(start = (y1, x1), end = (y2, x2))
        
        ret[rr, cc] = color_line
        #ret[rr, cc, 0] = 255

    return ret


class coco_base_(Dataset, data_loader.base):
    def __init__(self, cfg, data='train', transformer=None):
        
        self.anntype = cfg.ANNTYPE
        self.__data = data
        self.__name = cfg.NAME
        self.data_dir = cfg.PATH
        #self.n_class = cfg.NUM_CLASSES
        self.transformer = transformer
        self.pycocoloader(cfg)

        data_loader.base.__init__(self, cfg)
        Dataset.__init__(self)

        if self.anntype == 'bbox':
            print("get_bboxes")
            self.get_annotation = self.get_bboxes
            self.filter_BboxAnnotation = self.filter_BboxAnnotation_normal
            
            self.set_get_bbox(cfg)

        elif self.anntype == 'bbox_keyfilter':
            self.get_annotation = self.get_bboxes
            self.filter_BboxAnnotation = self.filter_BboxAnnotation_keyfilter
            self.th_keypoints = cfg.FILTER_KEYPOINT
            self.set_get_bbox(cfg)

        elif self.anntype == 'keypoints':
            print("get_keypoints")
            self.get_annotation = self.get_keypoints
            self.filter_num_joints = cfg.FILTER_NUM_JOINTS
            self.crop_type = cfg.CROP_TYPE
            self.crop_offset = cfg.CROP_OFFSET
            self.set_get_bbox(cfg)



        self.cropped_coordinate = cropped
        self.iscrowd_exist = True

        self.fmt_bbox = "COCO"
        self.fmt_keypoint = "COCO"
        self.cvt_keypoint_coco2mpii = []


    def set_get_bbox(self, cfg):

        if cfg.BBOX_CORRECTION == "normal":
            self._get_bbox = self._get_bbox_normal
            self.w_coeff = 1.0
            self.h_coeff = 1.0

        elif cfg.BBOX_CORRECTION == "enlarge_person":
            self._get_bbox = self._get_bbox_enlarge_person
            self.w_coeff = cfg.BBOX_CORRECTION_W
            self.h_coeff = cfg.BBOX_CORRECTION_H
        elif cfg.BBOX_CORRECTION == "enlarge_random":
            self._get_bbox = self._get_bbox_enlarge_random
            self.w_coeff = cfg.BBOX_CORRECTION_W
            self.h_coeff = cfg.BBOX_CORRECTION_H
        else:
            raise ValueError(cfg.BBOX_CORRECTION)


    def initialize_loader(self):
        self.get_ids_image()
    
    def set_ids_function(self, key4func, func):
        self.ids_funcs[key4func] = func

    def get_prefix(self, v):
        __prefix = 'instances'
        if v == "bbox":
            __prefix = 'instances'
        elif v == "keypoints":
            __prefix = 'person_keypoints'
        elif v == "bbox_keyfilter":
            __prefix = 'person_keypoints'
        elif v == "captions":
            __prefix = 'captions'
        else:
            raise ValueError("choose from [bbox, keypoints, captions]")
        return __prefix
    
    def get_image_dir(self):

        ret = self.data_dir + '/images/' + self.__data + self.__name + '/'
        return ret

    def get_anns_dir(self):

        temp = self.__data + self.__name
        ret = '%sannotations/%s_%s.json'%(self.data_dir, self.prefix, temp)
        return ret


    def pycocoloader(self, cfg):
        
        self.prefix = self.get_prefix(cfg.ANNTYPE) #instances, person_keypoints, captions
        self.img_dir = self.get_image_dir()
        self.annfname = self.get_anns_dir()
        print(self.annfname)
        self.coco = COCO(self.annfname)
        #self.ids_img = []
        self.ids = []
        self.map_catID = {}


    def get_ids_image(self):
        raise NotImplementedError()

    def categories(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms=[cat['name'] for cat in cats]
        return nms

    def __len__(self):#length of mini-batches
        return self.num_data

    @property
    def num_data(self):
        return len(self.ids)

    def _get_bbox_normal(self, ann, h_img, w_img):

        #xywh (x_center, y_center, width, height)
        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        w = float(ann['bbox'][2])
        h = float(ann['bbox'][3])
        id_cat = self.map_catID[int(ann['category_id'])]

        return [x1, y1, w, h, id_cat]

    def _get_bbox_enlarge_base(self, ann, h_img, w_img, h_coeff, w_coeff):

        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        w = float(ann['bbox'][2] * w_coeff)
        h = float(ann['bbox'][3] * h_coeff)
            
        x1 = x1 + w * (1.0 - w_coeff) / 2.
        y1 = y1 + h * (1.0 - h_coeff) / 2.
        x1 = x1 if x1 > 0 else 0
        y1 = y1 if y1 > 0 else 0
        w_temp = w * w_coeff
        h_temp = h * h_coeff
        w = w_temp if (x1 + w_temp) < w_img - 1 else float(w_img - x1 - 1)
        h = h_temp if (y1 + h_temp) < h_img - 1 else float(h_img - y1 - 1)
        
        return [x1, y1, w, h]
    
    def _get_bbox_enlarge(self, ann, h_img, w_img):
        return self._get_bbox_enlarge_base(ann, h_img, w_img, self.h_coeff, self.w_coeff)

    def _get_bbox_enlarge_random(self, ann, h_img, w_img):
        
        multi_h = random.uniform(1.0, self.h_coeff)
        multi_w = random.uniform(1.0, self.w_coeff)
        return self._get_bbox_enlarge_base(ann, h_img, w_img, multi_h, multi_w)

    def _get_bbox_enlarge_person(self, ann, h_img, w_img):

        id_cat = self.map_catID[int(ann['category_id'])]
        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        if id_cat == 0:
            w = float(ann['bbox'][2] * self.w_coeff)
            h = float(ann['bbox'][3] * self.h_coeff)
            x1 = x1 + w * (1.0 - self.w_coeff) / 2.
            y1 = y1 + h * (1.0 - self.h_coeff) / 2.
            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            w = w if (x1 + w) < w_img - 1 else float(w_img - x1 - 1)
            h = h if (y1 + h) < h_img - 1 else float(h_img - y1 - 1)
        else:
            w = float(ann['bbox'][2])
            h = float(ann['bbox'][3])
        return [x1, y1, w, h, id_cat]


    def filter_BboxAnnotation_normal(self, a):

        if self.iscrowd_exist == True:
            if (len(a['bbox']) > 0) and (int(a['iscrowd']) == 0) and (int(a['category_id']) in self.map_catID.keys()):
                return True
            else:
                return False
        else:
            if (len(a['bbox']) > 0) and (int(a['category_id']) in self.map_catID.keys()):
                return True
            else:
                return False

    def filter_BboxAnnotation_keyfilter(self, a):

        keypoints = np.array(a['keypoints']).reshape((-1, 3))
        num_keypoints = (keypoints[:, 2] > 0.1).sum()

        #print(num_keypoints)
        if num_keypoints >= self.th_keypoints:
            if (len(a['bbox']) > 0) and (int(a['iscrowd']) == 0) and (int(a['category_id']) in self.map_catID.keys()):
                return True
            else:
                return False
        else:
            return False

    def get_img(self, img_id):
        #img_id = self.coco.getImgIds(imgIds=_id)
        #img_id = self.ids[_id]
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = self.img_dir + img_name
        if os.path.exists(img_path) == False:
            return None
        else:
            img = io.imread(img_path)
            img_shape = img.shape
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3)) #(y, x, c)
            return img


    def get_bboxes(self, img, anns, transformer):
        
        
        labels = [self._get_bbox(a, img.shape[0], img.shape[1]) for a in anns if self.filter_BboxAnnotation(a)]
        if len(labels) > 0:
            labels = [ls for ls in labels if (ls[2] > 5.) and (ls[3] > 5.)]
            
        if len(labels) > 0:
            
            labels = np.array(labels)
            if transformer is not None:
                augmented = transformer(image=img, bboxes = labels[:, 0:4], category_id = labels[:, 4])
            else:
                augmented = dict(image=img, bboxes = labels[:, 0:4], category_id = labels[:, 4])

            #filter against the bbox size
            #temp = [list(b) + [c] for b, c in zip(augmented["bboxes"], augmented["category_id"]) if (b[2] > 8.) and (b[3] > 8.)]
            itr = enumerate(augmented["bboxes"])
            index_temp = [i for i, b in itr \
                if (b[2] > 8.) and (b[3] > 8.) and (b[0] >= 0.) and (b[1] >= 0.) and (b[0] + b[2] <= augmented["image"].shape[1]) and (b[1] + b[3] <= augmented["image"].shape[0])]
            
            if len(index_temp) > 0:
                temp = [list(augmented["bboxes"][loop]) + [augmented["category_id"][loop]] for loop in index_temp]
                temp = np.array(temp)
                augmented["bboxes"] = temp[:, 0:4]
                augmented["category_id"] = temp[:, 4]
            else:
                augmented = {"image":augmented["image"], "bboxes":[], "category_id":[]}

        else:
            if transformer is not None:
                augmented = transformer(image=img, bboxes = [], category_id = [])
                augmented = {"image":augmented["image"], "bboxes":[], "category_id":[]}
            else:
                augmented = {"image":img, "bboxes":[], "category_id":[]}

        return augmented


    def get_joints_new(self, joint, i_person):
        """
        """
        joints = np.array(joint['keypoints']).reshape((-1, 3)) # x, y, c
        joints_num = np.array(range(joints.shape[0]))[:, np.newaxis] # joint number
        joints = np.concatenate((joints, joints_num), axis = 1) # x, y, c, j
        joints_new = joints[joints[:, 2] > 0] # conf=0 is so many
        if joints_new.shape[0] < 3:
            return None
        else:
            temp = (joints_new, np.full((joints_new.shape[0], 1), i_person)) 
            return np.concatenate(temp, axis=1) # x, y, confidence, joint, person


    def get_keypoints(self, img, anns, transformer):
        
        if len(anns) == 0:
            augmented = {"image":img, "keypoints":[], "category_id":[]}

        else:
            joints = [self.get_joints_new(a, i) for i, a in enumerate(anns)]
            joints = [j for j in joints if j is not None]

            if len(joints) == 0:
                augmented = {"image":img, "keypoints":[], "category_id":[]}
            elif len(joints) == 1:
                joints = joints[0]
                #pass
            else:
                joints = np.concatenate([*joints], axis=0)
            
            if transformer is not None:
                #print(joints)
                augmented = transformer(image=img, keypoints=joints)
            else:
                augmented = {"image":img, "keypoints":joints}
            
        return augmented


    def __getitem__(self, i):
        
        #print(i)
        if self.map_catID["id"] == "img":
            return self.__getitem__img(i)

        elif self.map_catID["id"] == "ann+img":
            return self.__getitem__ann_img(i)

    def __getitem__img(self, i):
        """
        """
        img_id = self.ids[i]

        img = self.get_img(img_id)
        if img is None:
            data = {'image': None}
            return data
        else:
            img_shape = img.shape

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        data = self.get_annotation(img, anns, self.transformer)
        data["id_img"] = img_id
        # data["imsize"] = (img_shape[1], img_shape[0]) # width, height
        data["imsize"] = (img_shape[0], img_shape[1]) # height, width 

        return data


    def __getitem__ann_img(self, i):
        """
        #https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_keypoints.ipynb
        """
        
        img_id = self.ids[i][1]
        img = self.get_img(img_id)
        if img is None:
            data = {'image': None}
            return data
        else:
            img_shape = img.shape

        ann_id = self.ids[i][0]
        anns = self.coco.loadAnns(ann_id)
        if self.iscrowd_exist == True:
            if int(anns[0]['iscrowd']) != 0:
                return {"image":img, "bboxes":[], "category_id":[], "keypoints":[[]]}

        data = self.get_annotation(img, anns, None) # augmented = {"image":img, "keypoints":joints}
        joints_new = np.array(data["keypoints"])
        if len(data["keypoints"]) == 0:
            return {"image":None, "bboxes":[], "category_id":[], "keypoints":[[]]}

        bbox4keys = self._get_bbox(anns[0], img.shape[0], img.shape[1])
        _x_min = int(bbox4keys[0])
        _y_min = int(bbox4keys[1])
        _x_max = int(_x_min + bbox4keys[2])
        _y_max = int(_y_min + bbox4keys[3])
        center = np.array([_x_min, _y_min])
        scale = np.array([(_x_max - _x_min), (_y_max - _y_min)]) 

        if self.cropped_coordinate == True:
            crop = Compose([Crop(x_min=_x_min, y_min=_y_min, x_max=_x_max, y_max=_y_max, always_apply=True)],\
                            keypoint_params=A.KeypointParams(format='xy'))
            img_cropped = crop(image=img, keypoints=joints_new)

        else:
            img_cropped = {"image":img, "keypoints":joints_new}

        if self.transformer is not None:
            #augmented = self.transformer(image=img_cropped["image"], keypoints=img_cropped["keypoints"])
            augmented = self.transformer(img_cropped["image"], img_cropped["keypoints"])
            augmented["keypoints"] = [augmented["keypoints"]]
            augmented["id_img"] = img_id
            augmented["center"] = center
            augmented["scale"] = scale

        else:
            augmented = {"image":img_cropped["image"], \
                         "keypoints":[img_cropped["keypoints"]],\
                         "id_img":img_id,
                         "center":None,
                         "scale":None}
            
        return augmented



class coco_base_specific_(coco_base_):

    def __init__(self, cfg, data='train', transformer = None):
        
        super(coco_base_specific_, self).__init__(cfg, data, transformer)
        self.ids_funcs = {}
        self.set_ids_function("all", func_all)
        self.set_ids_function("commercial", func_commercial)
        self.set_ids_function("all_pattern1", func_all_pattern1)
        
        self.set_ids_function("custom1", func_custom1)
        self.set_ids_function("vehicle", func_vehicle)
        self.set_ids_function("vehicle_all", func_vehicle_all)
        self.set_ids_function("person", func_person)
        self.set_ids_function("person_commercial", func_person_commercial)
        self.set_ids_function("personANDothers", func_personANDothers)
        self.set_ids_function("personANDothers2", func_personANDothers2)
        self.set_ids_function("personANDothers2_commercial", func_personANDothers2_commercial)
        
        self.set_ids_function("keypoint_image", func_keypoint_image)
        self.set_ids_function("keypoints", func_keypoints)
        self.set_ids_function("keypoint2image_commercial", func_keypoint2image_commercial)
        self.ids_image_form = cfg.IDS #'all', ''
        self.initialize_loader()

    @property
    def ids_image_form(self):
        return self.__ids_image_form

    @ids_image_form.setter
    def ids_image_form(self, v):
        print(v)
        print(self.ids_funcs.keys())
        if v in self.ids_funcs.keys():
            self.__ids_image_form = v
        else:
            self.__ids_image_form = "commercial"
            cmt = ""
            for loop in self.ids_funcs.keys():
                cmt += loop + ", "
            raise ValueError("choose from " + cmt)

    def get_ids_image(self):

        key = self.ids_image_form
        if key in self.ids_funcs.keys():
            #self.ids_img, self.map_catID = self.ids_funcs[key](self.coco)
            self.ids, self.map_catID, \
                self.map_invcatID, self.categories_new = self.ids_funcs[key](self.coco)
            
            #if self.map_catID["id"] == "img":
            #    self.__getitem__ = self.__getitem__img
            #elif self.map_catID["id"] == "ann+img":
            #    self.__getitem__ = self.__getitem__ann_img

        else:
            raise ValueError("set ids_image_form correctly")


class coco_original(coco_base_specific_):
    name = 'original'
    def __init__(self, cfg, data='train', transformer=None):
        
        self.__data = data #
        self.__name = original_name #
        self.year = year #
        
        super(coco_original, self).__init__(cfg, data, transformer)
        # super(coco_original, self).__init__(cfg, data, transformer, name=original_name, cropped=cropped)

        #self.pycocoloader(cfg)

    def get_image_dir(self):

        ret = self.data_dir + '/images/' + self.__data + self.year + '/'
        print(ret)
        return ret

    def get_anns_dir(self):

        dataName = self.__data + self.__name
        ret = '%sannotations/%s_%s.json'%(self.data_dir, self.prefix, dataName)
        print(ret)
        return ret




def x1y1wh_to_xywh(label):
    ret = np.copy(label)
    ret[:, 0:2] += ret[:, 2:4] / 2.
    return ret

def check_loader(cfg, coco, compose=None):
    
    print("Check coco dataloader")
    cfg.IDS = 'vehicle'
    #cfg.IDS = 'vehicle'
    data_type = 'train'
    #data_type = 'val'

    #loader = coco(cfg, 'val', compose)
    data_ = coco2017_(cfg, data_type, compose)
    data_.initialize_loader()
    
    #loader = DataLoader(data_, batch_size=cfg.BATCHSIZE, shuffle=False, sampler=None,
    #                    batch_sampler=None, num_workers=12, collate_fn=collate_fn,
    #                    pin_memory=False, drop_last=False, timeout=0,
    #                    worker_init_fn=None)
    #loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
    #                    shuffle=True, num_workers=12, collate_fn=data_.collate_fn)
    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_.collate_fn)
    #loader = DataLoader(data_, batch_size=cfg.BATCHSIZE, num_workers=12, collate_fn=collate_fn)

    s = 0
    object_num = np.zeros(80)
    for batch_idx, (imgs, targets_list) in enumerate(loader):
    #for batch_idx, data in enumerate(loader):

        print ("load and agument:{0}".format(time.time() - s) + "[sec]")
        print(np.array(imgs).shape)
        #print(np.array(img).shape, np.array(target).shape)
        for targets in targets_list[0]:
            for t in targets:
                object_num[int(t[4])] += 1
        #print(target)
        d = (batch_idx+1)/len(loader) * 100
        print('[{} / {}({:.1f}%)]'.format(batch_idx+1, len(loader), d))
        
        s = time.time()

    path = "img_loader/dataset/"
    data_type_ = data_type + "_" + cfg.IDS
    plot_num_dataset(path, data_type_, [object_num], [data_type])
    np.save(path + "data_"  + data_type_  + ".npy", object_num)



def check_bbox(cfg, coco, compose=None):

    print("Check coco BBox and data augmentations")

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    #compose = None
    cfg.IDS = 'all'
    #cfg.IDS = 'vehicle'
    data_type = 'val'

    data_ = coco2017_(cfg, data_type, compose)
    data_.initialize_loader()
    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_.collate_fn)
    #data.form = "x1y1whc"

    path = "./dataset/temp2/"
    operator.remove_files(path)
    operator.make_directory(path)

    pl.figure()
    for i, (img, target) in enumerate(loader):
        #print(img.shape, target.shape)
        if i > 3:
            break
        
        for ii, (c, t) in enumerate(zip(img, target[0])):

            fname = path + str(i*32 + ii) + ".jpg"
            print(fname)
            c_box = draw_box(c, t, "xywhc")
            pl.clf()
            pl.imshow(c_box)
            pl.savefig(fname)


def plot_num_dataset(path, data_type, object_num, content):
    
    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as plt

    print(object_num)
    plt.figure()
    #label = ["Apple", "Banana", "Orange", "Grape", "Strawberry"]
    #labels = data_test.dataset.categories()
    #plt.bar(range(1, len(object_num)+1),  object_num, tick_label=labels, align="center")
    #plt.bar(range(1, len(object_num)+1),  object_num,  align="center")
    for o in object_num:
        plt.bar(range(1, len(o)+1), o,  align="center")
        print(np.argwhere(o < 1)) 
        print(np.sum(o))

    plt.legend(content)
    plt.grid()
    plt.yscale('log')
    plt.ylim([1e0, 4e5])
    plt.savefig(path + "data_"  + data_type  + ".png")
    # 2, 16, 21
    # 2, 16, 21



def check_keypoints(cfg):

    print("Check COCO keypoints")
    import augmentator
    #from keypoint_detector.config.mspn import get_augmentator_val

    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as plt

    cfg = edict()
    cfg.NAME = "COCO2017"
    cfg.PATH = '/data/public_data/COCO2017/'
    cfg.ANNTYPE = 'keypoints'

    #if True:
    if False:
        cfg.IDS = 'keypoint_image'
        cfg.FORM = "xyc"
    else:
        cfg.IDS = 'keypoints'
        cfg.FORM = "xycb"

    cfg.FILTER_NUM_JOINTS = 3
    cfg.BATCHSIZE = 16
    cfg.CROP_TYPE = 'fix'
    #cfg.CROP_TYPE = 'random'
    cfg.CROP_OFFSET = 3
    dtype = "val"

    image_size = (256, 192)
    # Crop 80 - 100% of image
    crop_min = image_size[0]*80//100
    crop_max = image_size[0]
    crop_min_max = (crop_min, crop_max)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10

    fmt = "coco"
    #compose = get_augmentator_val((416, 416))
    compose = None

    data_ = coco2017_(cfg, dtype, compose)
    data_.initialize_loader()
    
    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_.collate_fn)

    plt.figure()
    path = "./models/temp/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        import shutil
        shutil.rmtree(path)
        os.makedirs(path)

    for i, (imgs, anns) in enumerate(loader):
        
        print(np.array(imgs[0]).shape)
        for ii, img in enumerate(imgs):

            fname = path + str(i*cfg.BATCHSIZE + ii) + ".jpg"
            plt.clf()
            plt.imshow(img)
            plt.savefig(fname)

        if i > 30:
            break


def check_licence(cfg, coco, compose):

    print("Check Licenses on images")

    for dtype in ["train", "val"]:
        data_ = coco(cfg, dtype, None)
        data_.initialize_loader()
        #data_.form = "icxywh_normalized"
        print(data_.coco.dataset.keys())
        license_num = {}
        for l in data_.coco.dataset['licenses']:
            license_num[l["id"]] = 0
        
        #for i in data_.indeces:
        for i in data_.coco.imgs.keys():
            id_license = data_.coco.imgs[i]['license']
            license_num[id_license] += 1

        print("------" + dtype + "------")
        #print(data_.coco.dataset['licenses'])
        for l in data_.coco.dataset['licenses']:
            print(l["id"])
            print(l["name"])
            print(l["url"])

        print(license_num)
        print("Numbers of data", len(data_.coco.imgs))


def check_annotations(cfg, coco, compose, year):
    
    print("Check Annotations on images")

    for dtype in ["train", "val"]:

        fname = cfg.PATH + "annotations/person_keypoints_" + dtype + year + ".json"
        cc = COCO(fname)
        
        # earn ids under free licences
        id_free_list = []
        for i in cc.getImgIds():
            id_license = cc.imgs[i]['license']
            if id_license >= 4:
                id_free_list.append(cc.imgs[i]['id'])
        
        N_body_free = 0
        N_body_ALL = 0
        ann_ids_free = cc.getAnnIds(np.array(id_free_list))
        for _id in ann_ids_free:
            if cc.anns[_id]['num_keypoints'] > 0:
                N_body_free += 1
        for _id in cc.getAnnIds():
            if cc.anns[_id]['num_keypoints'] > 0:
                N_body_ALL += 1
        print("------------" + dtype + "------------")
        print("N_imgs_free : ", len(id_free_list))
        print("N_body_ALL : ", N_body_ALL)
        print("N_body_free : ", N_body_free)
        print("-------------------------------------")

def check_cocoapi(cfg, coco, compose, year):
    
    #for dtype in ["train", "val"]:
    dtype  = "train"
    fname = cfg.PATH + "annotations/instances_" + dtype + year + ".json"
    #fname = cfg.PATH + "annotations/person_keypoints_" + dtype + year + ".json"
    cc = COCO(fname)
    cats = cc.loadCats(cc.getCatIds())
    nms=[cat['name'] for cat in cats]
    cat_id = [cat['id'] for cat in cats]
    sucat = [cat['supercategory'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    print(len(nms))
    print("nms", nms, len(nms))
    print(cc.getCatIds())
    print(cats[0].keys())
    print("cat_id", cat_id, len(cat_id))
    print("sucat", sucat, len(sucat))

    #nms = set([cat['supercategory'] for cat in cats])
    #print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catIds_1 = cc.getCatIds(catNms=['person','dog','skateboard'])
    imgIds_1 = cc.getImgIds(catIds=catIds_1)
    catIds_2 = cc.getCatIds(catNms=nms[30])
    imgIds_2 = cc.getImgIds(catIds=catIds_2)
    imgIds_3 = cc.getImgIds()

    print(len(imgIds_1), "imgIds_1")
    print(len(imgIds_2), "imgIds_2")
    print(len(imgIds_3), "imgIds_3")
    print(nms)

    print(catIds_2)
    annIds_2 = cc.getAnnIds(imgIds_2[0])
    for ann in cc.loadAnns(annIds_2):
        print(ann)

    for i, cat in enumerate(cats):
        print("-------", i, "-------", cat["name"])
        

        catIds = cc.getCatIds(catNms=cat["name"])
        imgIds = cc.getImgIds(catIds=catIds)

        print("catIds", catIds, catIds[-1], i)
        print("len(imgIds)", len(imgIds))

        #__map_catID[int(catIds[0])] = _loop
        #id_cat = self.map_catID[int(ann['category_id'])]
        #ret = [x1, y1, w, h, id_cat]
        #ann_ids = cc.getAnnIds(imgIds=imgIds[0])
        #anns = cc.loadAnns(ann_ids)
        #print(len(anns))
        #print((np.array(anns["bbox"])[:, 4] == i).shape)

    #print(cc.anns.items())


if __name__ == '__main__':

    print("start")
    #np.random.seed(1234)
    np.random.seed(9999)
    from easydict import EasyDict as edict
    cfg = edict()
    year = '2017'
    cfg.PATH = '/data/public_data/COCO2017/'
    cfg.ANNTYPE = 'bbox'
    cfg.IDS = 'all'
    cfg.BATCHSIZE = 30
    cfg.NUM_CLASSES = 80
    
    image_size = 416

    #fmt = "pascal_voc"
    fmt = "coco"

    print(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.PATH = sys.argv[2]

    if sys.argv[1] == "loader":
        check_loader(cfg)

    elif sys.argv[1] == "keypoints":
        cfg.FORM = "xycb"
        check_keypoints(cfg)

    elif sys.argv[1] == "bbox":
        cfg.FORM = "xywhc"
        check_bbox(cfg)

    elif sys.argv[1] == "test":
        path = "img_loader/dataset/"

        ids = "all"
        #data_type = "train"
        #data_type = "val"
        data_type = "train_val"
        object_num_train = np.load(path + "data_train_" + ids + ".npy")
        object_num_val = np.load(path + "data_val_" + ids + ".npy")
        plot_num_dataset(path, data_type, [object_num_train, object_num_val], ["train", "val"])

    elif sys.argv[1] == "cocoapi":
        check_cocoapi(cfg, coco, compose, year)
    #check_licence(cfg, coco, compose)
    #check_annotations(cfg, coco, compose, year)
   


    print("end")

#https://gist.github.com/Lexie88rus/b1f3a45f3e0e19c59c1795d7509d42a4

