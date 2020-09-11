
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
        return self.num_data // self.batchsize

    def categories(self):
        nms = [str(i+1) for i in range(self.n_class)]
        return nms


    @property
    def num_data(self):
        return len(self.ids_img)

    def get_bbox(self, ann):

        #print(self.map_catID, int(ann['category_id']), len(self.map_catID))
        included = int(ann['category_id']) in self.map_catID.keys()
        #print(included, int(ann['category_id']))
        if len(ann['bbox']) == 0:
            print("NO BOX")
            ret = None
        elif included == False:
            ret = None
        else:
            x1 = float(ann['bbox'][0])
            y1 = float(ann['bbox'][1])
            w = float(ann['bbox'][2])
            h = float(ann['bbox'][3])
            id_cat = self.map_catID[int(ann['category_id'])]
            #id_cat = int(ann['category_id'])
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
        __map_catID[int(catIds[0])] = _loop
    return __ret_img, __map_catID
    
def func_commercial(coco):
    __ret_img = []
    __map_catID = {}
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[0])] = _loop
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
        __map_catID[int(catIds[0])] = _loop
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
    cats = coco.loadCats(coco.getCatIds())
    nms = ["truck", "car", "bus"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[0])] = _loop
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
            x2 = int(t[0] * (img.shape[1] - 1) + w / 2)
            y2 = int(t[1] * (img.shape[0] - 1) + h / 2)

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


def check_bbox(cfg, coco, compose=None):

    print("Check coco BBox and data augmentations")

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    data = coco(cfg, 'check', compose)
    #data.form = "x1y1whc"

    path = "./dataset/temp/"
    operator.remove_files(path)
    operator.make_directory(path)

    pl.figure()
    for i, (img, target) in enumerate(data):
        #print(img.shape, target.shape)
        if i > 3:
            break
        
        for ii, (c, t) in enumerate(zip(img, target)):

            fname = path + str(i*32 + ii) + ".jpg"
            c_box = draw_box(c, t, data.form)
            pl.clf()
            pl.imshow(c_box)
            pl.savefig(fname)

def check_loader(cfg, coco, compose=None):

    print("Check coco dataloader")

    data_ = coco(cfg, 'val', compose)
    data_.initialize_loader()
    #data_.form = "icxywh_normalized"
    #print(data_train.coco.anns.keys())

    for batch_idx, (img, target) in enumerate(data_):
        print(np.array(img).shape, np.array(target).shape)
        #print(target)
        d = batch_idx/len(data_) * 100
        print('[{} / {}({:.1f}%)]'.format(batch_idx, len(data_), d))

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
    #for dtype in ["train", "val"]:
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
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    print(len(nms))

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


if __name__ == '__main__':

    print("start")
    #np.random.seed(1234)
    np.random.seed(9999)
    from easydict import EasyDict as edict
    from dataset.augmentator import get_compose, get_compose_keypoints

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
    cfg.NUM_CLASSES = 91
    
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

    compose = get_compose(crop_min_max, image_size, image_size, 
                          hue_shift, saturation_shift, value_shift, fmt)
    compose_keypoints = get_compose_keypoints(crop_min_max, image_size, image_size, 
                                    hue_shift, saturation_shift, value_shift, fmt)
    #compose = None

    print(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.PATH = sys.argv[2]

    if sys.argv[1] == "loader":
        cfg.FORM = "icxywh_normalized"
        check_loader(cfg, coco, compose)
    elif sys.argv[1] == "keypoints":
        cfg.FORM = "xyc"
        check_keypoints(cfg, coco, None)
    elif sys.argv[1] == "bbox":
        #cfg.FORM = "icxywh_normalized"
        cfg.FORM = "xywhc"
        
        check_bbox(cfg, coco, compose)
        
    #check_licence(cfg, coco, compose)
    #check_annotations(cfg, coco, compose, year)
    #check_cocoapi(cfg, coco, compose, year)


    print("end")





