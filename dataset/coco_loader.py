
import sys
import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import data_loader
import time

#class coco_base(data_loader.base_augmentation):
class coco_base(data_loader.base_augmentation0):

    def __init__(self, cfg, data='train', transformer = None, name="2017"):
        super(coco_base, self).__init__(cfg, transformer)
        
        self.__data = data
        self.data_dir = cfg.PATH
        self.n_class = cfg.NUM_CLASSES
        self.coco = None
        
        if self.anntype == 'bbox':
            self.get_annotation = self.get_bbox
        elif self.anntype == 'keypoints':
            self.get_annotation = self.get_keypoints

        self.pycocoloader(cfg, data, transformer, name)

    def initialize_loader(self):
        self.get_ids_image()
        self.__loop = 0
        self.indeces_batchs = self.get_indeces_batches()
        
    def get_prefix(self, v):
        __prefix = 'instances'
        if v == "bbox":
            __prefix = 'instances'
        elif v == "keypoints":
            __prefix = 'person_keypoints'
        elif v == "captions":
            __prefix = 'captions'
        else:
            raise ValueError("choose from [bbox, keypoints, captions]")
        return __prefix

    def get_dataName(self, data, name):
        if data == 'check':
            return 'val' + name
        else:
            return data + name

    def pycocoloader(self, cfg, data, transformer, name):
        
        prefix = self.get_prefix(cfg.ANNTYPE)
        dataName = self.get_dataName(data, name) # train2014
        self.img_dir = self.data_dir + '/images/' + dataName + '/'
        annfname = '%sannotations/%s_%s.json'%(self.data_dir, prefix, dataName)
        print(annfname)
        self.coco = COCO(annfname)
        self.ids_img = []
        self.map_catID = {}


    def get_ids_image(self):
        raise NotImplementedError()

    def __next__(self):
        if self.__loop >= len(self.indeces_batchs):
            self.initialize_loader()
            raise StopIteration()
        _ids = self.indeces_batchs[self.__loop]
        img_list, target_list = self.load(_ids)
        self.__loop += 1
        return (img_list, target_list)

    def __len__(self):#length of mini-batches
        #return self.num_data // self.batchsize
        return self.num_data

    def categories(self):
        nms = [str(i+1) for i in range(self.n_class)]
        return nms


    @property
    def num_data(self):
        return len(self.ids_img)

    def get_bbox(self, ann):

        #included = int(ann['category_id']) in self.map_catID.keys()
        #if len(ann['bbox']) == 0:
            #print("NO BOX")
            #ret = [None, None, None, None, None]
        #    ret = None
        #elif included == False:
            #print("not included")
            #ret = [None, None, None, None, None]
        #    ret = None
        #else:
            
        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        w = float(ann['bbox'][2])
        h = float(ann['bbox'][3])
        id_cat = self.map_catID[int(ann['category_id'])]
        ret = [x1, y1, w, h, id_cat]
        return ret

    def get_keypoints(self, ann):

        if ann['num_keypoints'] == 0:
            #print("NO keypoints")
            #print(ann)
            return None
        #ann['keypoints'].__len__ (21) -> joints.shape(7, 3)
        joints = np.array(ann['keypoints']).reshape((-1, 3))
        #print(ann['keypoints'], joints.shape)
        return joints


    def load(self, _ids_img):

        #https://pytorch.org/docs/stable/_modules/torchvision/datasets/coco.html#CocoDetection
        img_list = []
        target_list = []
        for i, _v in enumerate(_ids_img):
            #img
            img_id = self.ids_img[_v]
            img_name = self.coco.imgs[img_id]['file_name']
            img_path = self.img_dir + img_name
            if os.path.exists(img_path) == False:
                print("no file")
                continue
            else:
                #start = time.time()
                img = io.imread(img_path)
                #print ("load image:{0}".format(time.time() - start) + "[sec]")
                #print(np.max(img), np.min(img))#0-255

                if img.ndim == 2:
                    img = np.expand_dims(img, 2)
                    img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))

            #target
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            target = []

            if len(anns) == 0:
                continue

            #start = time.time()
            for ann in anns:
                #print(ann)
                ret = self.get_annotation(ann)
                #ret = self.get_bbox(ann)
                #print(ret)
                if ret is None:
                    continue
                else:
                    target.append(ret)
            #print ("Get annotation:{0}".format(time.time() - start) + "[sec]")

            if len(target) == 0:
                continue
            else:
                target_list.append(target)
                img_list.append(img)
        
        #start = time.time()
        img_list, target_list = self.transform(img_list, target_list)
        #print ("transform images and annotations:{0}".format(time.time() - start) + "[sec]")
        #print(target_list)
        return img_list, target_list


def func_all(coco):
    __ret_img = []
    __map_catID = {}
    __ret_img = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for _loop, cat in enumerate(nms):
        #print(cat)
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
    return __ret_img, __map_catID
    
def func_commercial(coco):
    __ret_img = []
    __map_catID = {}
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
    for _id in coco.getImgIds():
        id_license = coco.imgs[_id]['license']
        if id_license >= 4:
            #ret.append(cc.imgs[i]['id'])
            __ret_img.append(_id)
    return __ret_img, __map_catID
    
def func_custom1(coco):
    __ret_img = []
    __map_catID = {}
    _pickup = 200
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    #print("nms", nms, len(nms))
    for _loop, cat in enumerate(nms):
        #print(cat)
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        #print(_loop, catIds[0])
        #print(cat, len(imgIds))
        if len(imgIds) != 0:
            idx = np.random.choice(len(imgIds), _pickup)
        else:
            continue
        __ret_img += np.array(imgIds)[idx].tolist()
    return __ret_img, __map_catID

def func_vehicle(coco):
    __ret_img = []
    __map_catID = {}
    #cats = coco.loadCats(coco.getCatIds())
    nms = ["truck", "car", "bus"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __ret_img += imgIds
    return __ret_img, __map_catID


class coco_base_specific(coco_base):

    def __init__(self, cfg, data='train', transformer = None, name="2017"):
        super(coco_base_specific, self).__init__(cfg, data, transformer, name)
        
        self.ids_funcs = {}
        self.set_ids_function("all", func_all)

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
            self.ids_img, self.map_catID = self.ids_funcs[key](self.coco)
        else:
            raise ValueError("set ids_image_form correctly")

    def set_ids_function(self, key4func, func):
        self.ids_funcs[key4func] = func


class coco_specific(coco_base_specific):
    """
    Arg:
        cfg : configuration given by EasyDict.
             PATH, IDS, ANNTYPE, BATCHSIZE, NUM_CLASSES should be given. 
             
        data : "train", "val", "test", "check"

        transformer : Compose object from albumentations should be given.
                     image and its annotation will be augmentated by Compose.

        name : the COOC dataset year(str number) that you want to use.
    
    Example:
        from easydict import EasyDict as edict
        from dataset.augmentator import get_compose, get_compose_keypoints
        from albumentations import Compose
        from albumentations.augmentations.transforms import Resize

        cfg = edict()
        cfg.PATH = '/data/public_data/COCO2017/'
        cfg.ANNTYPE = 'bbox'
        cfg.BATCHSIZE = 32
        tf = Compose([Resize(image_height, image_width, p=1.0)],\
                      bbox_params={'format':format, 'label_fields':['category_id']})

        dataloader = coco_base(cfg, "train", tf, "2017")
        imgs, annotations, dataloader.__next__()
    """

    def __init__(self, cfg, data='train', transformer = None, name="2017"):
        
        super(coco_specific, self).__init__(cfg, data, transformer, name)
        
        self.set_ids_function("commercial", func_commercial)
        self.set_ids_function("custom1", func_custom1)
        self.set_ids_function("vehicle", func_vehicle)

        self.ids_image_form = cfg.IDS #'all', ''
        self.initialize_loader()


class coco2014(coco_specific):
    name = 'coco2014'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
        self.year = "2014"
        super(coco2014, self).__init__(cfg, data, transformer, name="2014")

class coco2017(coco_specific):
    name = 'coco2017'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
        self.year = "2017"
        super(coco2017, self).__init__(cfg, data, transformer, name="2017")
        


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

        x1 = np.clip(x1, 0, ret.shape[1] - 1)
        x2 = np.clip(x2, 0, ret.shape[1] - 1)
        y1 = np.clip(y1, 0, ret.shape[0] - 1)
        y2 = np.clip(y2, 0, ret.shape[0] - 1)
        print(x1, y1, x2, y2, ret.dtype, ret.shape)
        color_line = np.array([255, 0, 0], dtype=np.uint8)

        #rr, cc = rectangle(start = (y1, x1), end = (y2 - 1, x2 - 1))
        rr, cc = rectangle_perimeter(start = (y1, x1), end = (y2 - 2, x2 - 2))
        
        ret[rr, cc] = color_line
        #ret[rr, cc, 0] = 255

    return ret



def check_keypoints(cfg, coco, compose):

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    from skimage.draw import circle
    def make_keypoint(img, ann):
        
        ret = np.copy(img)
        print(ret.shape, ann.shape)
        for a in ann:
            #print(x, y)
            x = int(a[0])
            y = int(a[1])
            rr, cc = circle(y, x, 5, ret.shape)
            ret[rr, cc, :] = (255, 0, 0)
        return ret

    path = "./img_loader/dataset/temp/"
    operator.remove_files(path)
    operator.make_directory(path)
    cfg.ANNTYPE = 'keypoints'
    for dtype in ["train"]:
        _data = coco(cfg, dtype, compose)
        #_data = coco(cfg, dtype, None)
        for i, (img, anns) in enumerate(_data):
            #print(np.array(anns).shape, np.array(img).shape)
            for ii, (img_each, ann_each) in enumerate(zip(img, anns)):
                fname = path + str(i*32 + ii) + ".jpg"
                i_ret = make_keypoint(img_each, ann_each)
                pl.clf()
                pl.imshow(i_ret)
                pl.savefig(fname)
                
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


from torch.utils.data import DataLoader, Dataset

class coco_base_(Dataset, data_loader.base):
    def __init__(self, cfg, data='train', transformer = None, name="2017"):
        
        self.anntype = cfg.ANNTYPE
        self.__data = data
        self.data_dir = cfg.PATH
        self.n_class = cfg.NUM_CLASSES
        self.transformer = transformer
        self.pycocoloader(cfg, data, name)

        data_loader.base.__init__(self, cfg)
        Dataset.__init__(self)

        if self.anntype == 'bbox':
            self.get_annotation = self.get_bbox
        elif self.anntype == 'keypoints':
            self.get_annotation = self.get_keypoints

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
        elif v == "captions":
            __prefix = 'captions'
        else:
            raise ValueError("choose from [bbox, keypoints, captions]")
        return __prefix

    def get_dataName(self, data, name):
        if data == 'check':
            return 'val' + name
        else:
            return data + name

    def pycocoloader(self, cfg, data, name):
        
        prefix = self.get_prefix(cfg.ANNTYPE)
        dataName = self.get_dataName(data, name) # train2014
        self.img_dir = self.data_dir + '/images/' + dataName + '/'
        annfname = '%sannotations/%s_%s.json'%(self.data_dir, prefix, dataName)
        print(annfname)
        self.coco = COCO(annfname)
        self.ids_img = []
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
        return len(self.ids_img)
    

    def get_bbox(self, ann):

        #included = int(ann['category_id']) in self.map_catID.keys()
        #if len(ann['bbox']) == 0:
            #print("NO BOX")
            #ret = [None, None, None, None, None]
        #    ret = None
        #elif included == False:
            #print("not included")
            #ret = [None, None, None, None, None]
        #    ret = None
        #else:
            
        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        w = float(ann['bbox'][2])
        h = float(ann['bbox'][3])
        id_cat = self.map_catID[int(ann['category_id'])]
        ret = [x1, y1, w, h, id_cat]
        return ret

    def get_keypoints(self, ann):

        if ann['num_keypoints'] == 0:
            #print("NO keypoints")
            #print(ann)
            return None
        #ann['keypoints'].__len__ (21) -> joints.shape(7, 3)
        joints = np.array(ann['keypoints']).reshape((-1, 3))
        #print(ann['keypoints'], joints.shape)
        return joints

    def __getitem__(self, i):

        #print(i)
        img_id = self.ids_img[i]
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = self.img_dir + img_name
        if os.path.exists(img_path) == False:
            #print("no file")
            return {"image":None, "bboxes":[], "category_id":[]}
        else:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        if len(anns) == 0:
            #print("zero annotations")
            return {"image":img, "bboxes":[], "category_id":[]}
        else:
            labels = [self.get_bbox(a) for a in anns if (len(a['bbox']) > 0) and (int(a['category_id']) in self.map_catID.keys())]

        if len(labels) > 0:
            labels = np.array(labels)
            labels[:, 2:4] = np.clip(labels[:, 2:4], 0.1, 416 - 0.1)
            augmented = self.transformer(image=img, bboxes = labels[:, 0:4], category_id = labels[:, 4])
        else:
            #print("no labels")
            return {"image":img, "bboxes":[], "category_id":[]}

        return augmented

    def collate_fn(self, batch):

        images = [b["image"] for b in batch if len(b["bboxes"]) > 0]
        x1y1wh_trans = [b["bboxes"] for b in batch if len(b["bboxes"]) > 0]
        id_trans = [b["category_id"] for b in batch if len(b["bboxes"]) > 0]
        targets = [np.concatenate([x1y1wh_to_xywh(_x1y1wh), np.array(_id)[:, np.newaxis]], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]
        return images, targets

class coco_base_specific_(coco_base_):

    def __init__(self, cfg, data='train', transformer = None, name="2017"):
        super(coco_base_specific_, self).__init__(cfg, data, transformer, name=name)
        self.ids_funcs = {}
        self.set_ids_function("all", func_all)
        self.set_ids_function("commercial", func_commercial)
        self.set_ids_function("custom1", func_custom1)
        self.set_ids_function("vehicle", func_vehicle)
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
            self.ids_img, self.map_catID = self.ids_funcs[key](self.coco)
        else:
            raise ValueError("set ids_image_form correctly")


class coco2017_(coco_base_specific_):
    name = 'coco2017'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
        self.year = "2017"
        super(coco2017_, self).__init__(cfg, data, transformer, name="2017")

class coco2014_(coco_base_specific_):
    name = 'coco2014'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
        self.year = "2014"
        super(coco2014_, self).__init__(cfg, data, transformer, name="2014")


def x1y1wh_to_xywh(label):
    ret = np.copy(label)
    ret[:, 0:2] += ret[:, 2:4] / 2.
    return ret

#def collate_fn(batch):

#    images = [b["image"] for b in batch if len(b["bboxes"]) > 0]
#    x1y1wh_trans = [b["bboxes"] for b in batch if len(b["bboxes"]) > 0]
#    id_trans = [b["category_id"] for b in batch if len(b["bboxes"]) > 0]
#    targets = [np.concatenate([x1y1wh_to_xywh(_x1y1wh), np.array(_id)[:, np.newaxis]], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]
#    return images, targets


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
        #print(np.array(img).shape, np.array(target).shape)
        for targets in targets_list:
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

    data = coco(cfg, 'check', compose)
    cfg.IDS = 'all'
    #cfg.IDS = 'vehicle'
    data_type = 'val'

    data_ = coco2017_(cfg, data_type, compose)
    data_.initialize_loader()
    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=data_.collate_fn)
    #data.form = "x1y1whc"

    path = "./dataset/temp/"
    operator.remove_files(path)
    operator.make_directory(path)

    pl.figure()
    for i, (img, target) in enumerate(loader):
        #print(img.shape, target.shape)
        if i > 3:
            break
        
        for ii, (c, t) in enumerate(zip(img, target)):

            fname = path + str(i*32 + ii) + ".jpg"
            print(fname)
            c_box = draw_box(c, t, data.form)
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
    
if __name__ == '__main__':

    print("start")
    #np.random.seed(1234)
    np.random.seed(9999)
    from easydict import EasyDict as edict
    from dataset.augmentator import get_compose_resize, get_compose, get_compose_resize2, get_compose_resize4
    from dataset.augmentator import get_compose_keypoints
    from dataset.augmentator import get_compose_resize5

    cfg = edict()
    #if True:
    if False:
        year = '2014'
        cfg.PATH = '/data/public_data/COCO2014/'
        coco = coco2014
    else:
        year = '2017'
        cfg.PATH = '/data/public_data/COCO2017/'
        coco = coco2017

    cfg.ANNTYPE = 'bbox'
    cfg.IDS = 'all'
    cfg.BATCHSIZE = 30
    #cfg.NUM_CLASSES = 91
    cfg.NUM_CLASSES = 80
    
    # Image size for YOLO
    image_size = 416
    # Crop 80 - 100% of image
    crop_min = image_size*80//100
    crop_max = image_size
    crop_min_max = (crop_min, crop_max)
    # HSV shift limits
    hue_shift = 10
    saturation_shift = 10
    value_shift = 10

    #fmt = "pascal_voc"
    fmt = "coco"

    
    compose_keypoints = get_compose_keypoints(crop_min_max, image_size, image_size, 
                                    hue_shift, saturation_shift, value_shift, fmt)

    #compose = get_compose_resize( image_size, image_size, fmt)
    #compose = get_compose_resize2(crop_min_max, image_size, image_size, 
    #                              hue_shift, saturation_shift, value_shift, fmt)
    #compose = get_compose_resize4(crop_min_max, image_size, image_size, 
    #                              hue_shift, saturation_shift, value_shift, fmt)
    compose = get_compose_resize5(crop_min_max, image_size, image_size, 
                                  hue_shift, saturation_shift, value_shift, fmt)
    #compose = None


    print(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.PATH = sys.argv[2]

    if sys.argv[1] == "loader":
        #cfg.FORM = "icxywh_normalized"
        cfg.FORM = "xywhc"
        check_loader(cfg, coco, compose)

    elif sys.argv[1] == "keypoints":
        cfg.FORM = "xyc"
        check_keypoints(cfg, coco, None)
    elif sys.argv[1] == "bbox":
        #cfg.FORM = "icxywh_normalized"
        cfg.FORM = "xywhc"
        
        check_bbox(cfg, coco, compose)

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

