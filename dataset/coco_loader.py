
import sys
import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import data_loader
import time


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
    __map_catID["id"] = "img"
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
    __map_catID["id"] = "img"
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
    __map_catID["id"] = "img"
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
    __map_catID["id"] = "img"
    return __ret_img, __map_catID


def func_vehicle_all(coco):
    __ret_img = []
    __map_catID = {}
    #cats = coco.loadCats(coco.getCatIds())
    nms = ["truck", "car", "bus", "bicycle", "motorcycle", "airplane", "train", "boat"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __ret_img += imgIds
    __map_catID["id"] = "img"
    return __ret_img, __map_catID

def func_keypoints(coco):
    __ret_ann = []
    __map_catID = {}
    
    for aid, ann in coco.anns.items():
        if ann['num_keypoints'] > 0:
            #__ret_img.append(ann['image_id'])
            #__ret_ann.append(aid)
            __ret_ann.append([aid, ann['image_id']])

    __map_catID["id"] = "ann+img"
    return __ret_ann, __map_catID



def func_person(coco):
    __ret_img = []
    __map_catID = {}
    nms = ["person"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __ret_img += imgIds
    __map_catID["id"] = "img"
    return __ret_img, __map_catID


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



def check_keypoints(cfg, coco, compose):

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    from skimage.draw import circle
    def make_keypoint(img, ann):
        
        ret = np.copy(img)
        for a in ann:
            for xy in a:
                #print(xy)
                x = int(xy[0])
                y = int(xy[1])
                rr, cc = circle(y, x, 5, ret.shape)
                #print(xy)
                if xy[2] == 1:
                    _color = (0, 64, 0)
                elif xy[2] == 2:
                    _color = (255, 0, 0)
                else:
                    continue
                ret[rr, cc, :] = _color
        return ret

    path = "./dataset/temp3/"
    operator.remove_files(path)
    operator.make_directory(path)
    cfg.ANNTYPE = 'keypoints'
    #cfg.IDS = 'all'
    cfg.IDS = 'keypoints'

    for dtype in ["val"]:

        data_ = coco2017_(cfg, dtype, compose)
        #data_ = coco2017_(cfg, dtype, None)
        data_.initialize_loader()
        loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                            shuffle=False, num_workers=2, collate_fn=data_.collate_fn)

        for i, (imgs, anns) in enumerate(loader):
            #print(np.array(imgs).shape, len(anns))
            #print(np.array(anns).shape, np.array(img).shape)
            for ii, (img_each, ann_each) in enumerate(zip(imgs, anns[0])):

                #print(ii)
                #print(ann_each, np.array(ann_each).shape)

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

    #print(cc.anns.items())


from torch.utils.data import DataLoader, Dataset
import albumentations as A 
from albumentations import Compose
from albumentations.augmentations.transforms import Crop

class coco_base_(Dataset, data_loader.base):
    def __init__(self, cfg, data='train', transformer=None, name="2017", cropped=True):
        
        self.anntype = cfg.ANNTYPE
        self.__data = data
        self.data_dir = cfg.PATH
        #self.n_class = cfg.NUM_CLASSES
        self.transformer = transformer
        self.pycocoloader(cfg, data, name)

        data_loader.base.__init__(self, cfg)
        Dataset.__init__(self)

        if self.anntype == 'bbox':
            self.get_annotation = self.get_bboxes
        elif self.anntype == 'keypoints':
            self.get_annotation = self.get_keypoints
        self.cropped_coordinate = cropped

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
        
        prefix = self.get_prefix(cfg.ANNTYPE) #instances, person_keypoints, captions
        dataName = self.get_dataName(data, name) # train2014
        self.img_dir = self.data_dir + '/images/' + dataName + '/'
        annfname = '%sannotations/%s_%s.json'%(self.data_dir, prefix, dataName) #
        print(annfname)
        self.coco = COCO(annfname)
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

    def _get_bbox(self, ann):

        #xywh (x_center, y_center, width, height)
        x1 = float(ann['bbox'][0])
        y1 = float(ann['bbox'][1])
        w = float(ann['bbox'][2])
        h = float(ann['bbox'][3])
        id_cat = self.map_catID[int(ann['category_id'])]

        return [x1, y1, w, h, id_cat]
        
    def get_bboxes(self, img, anns):
        
        #for a in anns:
        #    if a['iscrowd'] == 1:
        #        print("iscrowd")

        labels = [self._get_bbox(a) for a in anns \
                  if (len(a['bbox']) > 0) and (int(a['iscrowd']) == 0) and (int(a['category_id']) in self.map_catID.keys())]
                  #if (len(a['bbox']) > 0)  and (int(a['category_id']) in self.map_catID.keys())]

        if len(labels) > 0:
            labels = [ls for ls in labels if (ls[2] > 5.) and (ls[3] > 5.)]
            
        if len(labels) > 0:
            
            #ret = [x1, y1, w, h, id_cat]
            
            labels = np.array(labels)
            
            #labels[:, 2:4] = np.clip(labels[:, 2:4], 0.1, 416 - 0.1)
            if self.transformer is not None:
                augmented = self.transformer(image=img, bboxes = labels[:, 0:4], category_id = labels[:, 4])
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
                augmented = {"image":img, "bboxes":[], "category_id":[]}

            #https://github.com/aleju/imgaug

        else:
            #print("no labels")
            augmented = {"image":img, "bboxes":[], "category_id":[]}

        return augmented

    def get_keypoints(self, img, anns):
        
        #print(anns)
        if len(anns) == 0:
            augmented = {"image":img, "keypoints":[], "category_id":[]}

        else:
            #joints = [np.array(a['keypoints']).reshape((-1, 3)) for a in anns]
            joints = [np.array(a['keypoints']).reshape((-1, 3)) for a in anns if np.sum(np.array(a['keypoints'])) != 0]
            
            #center, scale = self._bbox_to_center_and_scale(bbox)
            #center = d['center']
            #scale = d['scale']

            #joints = np.array(anns['keypoints']).reshape((-1, 3))
            #joints_vis = joints[:, -1].reshape((-1, 1))
            if self.transformer is not None:
                augmented = self.transformer(image=img, keypoints=joints)
                #augmented["center"] = anns["center"]
                #augmented["scale"] = anns["scale"]
                #return img, score, center, scale, img_id
            else:
                #augmented = {"image":img, "keypoints":joints}
                augmented = {"image":img, "keypoints":joints}
            
        return augmented

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        #scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
        #        dtype=np.float32)

        #return center, scale

    def __getitem__(self, i):

        if self.map_catID["id"] == "img":
            return self.__getitem__img(i)

        elif self.map_catID["id"] == "ann+img":
            return self.__getitem__ann_img(i)

        
    def __getitem__img(self, i):

        #if self.map_catID["id"] == "img":
        #if self.map_catID["id"] == "ann":
        #if self.map_catID["id"] == "ann+img":
        #    img_id = self.ids[i][1]

        img_id = self.ids[i]
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = self.img_dir + img_name
        if os.path.exists(img_path) == False:
            return {"image":None, "bboxes":[], "category_id":[], "keypoints":[]}

        else:
            img = io.imread(img_path)
            img_shape = img.shape
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3)) #(y, x, c)
            
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        data = self.get_annotation(img, anns)
        data["id_img"] = img_id
        data["imsize"] = (img_shape[1], img_shape[0]) # width, height
        #data["image"] = data["image"].tolist()

        return data

    def __getitem__ann_img(self, i):
        #https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_keypoints.ipynb

        ann_id = self.ids[i][0]
        anns = self.coco.loadAnns(ann_id)
        if int(anns[0]['iscrowd']) != 0:
            return {"image":None, "bboxes":[], "category_id":[], "keypoints":[]}

        img_id = self.ids[i][1]
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = self.img_dir + img_name
        if os.path.exists(img_path) == False:
            return {"image":None, "bboxes":[], "category_id":[], "keypoints":[]}

        else:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))

        
        #center, scale = self._bbox_to_center_and_scale(anns[0]['bbox'])

        #bbox = anns[0]["bbox"]
        joints = np.array(anns[0]['keypoints']).reshape((-1, 3))
        joints_num = np.array(range(joints.shape[0]))[:, np.newaxis]
        joints = np.concatenate((joints, joints_num), axis = 1)
        joints_new = joints[joints[:, 2] > 0]

        # going to crop images so that ALL keypoints is on cropped image
        #ofs = 20
        ofs = 30
        #ofs = 40
        #ofs = 80
        _x_min = int(max([np.min(joints_new[:, 0]) - ofs, 0]))
        _y_min = int(max([np.min(joints_new[:, 1]) - ofs, 0]))
        _x_max = int(min([np.max(joints_new[:, 0]) + ofs, img.shape[1]-1]))
        _y_max = int(min([np.max(joints_new[:, 1]) + ofs, img.shape[0]-1]))

        center = np.array([_x_min, _y_min])
        scale = np.array([(_x_max - _x_min), (_y_max - _y_min)]) 

        #print("self.cropped_coordinate : ", self.cropped_coordinate)
        if self.cropped_coordinate == True:
            crop = Compose([Crop(x_min=_x_min, y_min=_y_min, x_max=_x_max, y_max=_y_max, always_apply=True)],\
                            keypoint_params=A.KeypointParams(format='xy'))
            img_cropped = crop(image=img, keypoints=joints)

        else:
            img_cropped = {"image":img, "keypoints":joints}

        if self.transformer is not None:
            augmented = self.transformer(image=img_cropped["image"], keypoints=img_cropped["keypoints"])
            augmented["keypoints"] = [augmented["keypoints"]]
            augmented["id_img"] = img_id
            augmented["center"] = center
            augmented["scale"] = scale

        else:
            img_cropped["keypoints"] = [img_cropped["keypoints"]]
            augmented = {"image":img_cropped["image"], \
                         "keypoints":img_cropped["keypoints"],
                         "id_img":img_id,
                         "center":None,
                         "scale":None}
            
        return augmented

class coco_base_specific_(coco_base_):

    def __init__(self, cfg, data='train', transformer = None, name="2017", cropped=True):
        
        super(coco_base_specific_, self).__init__(cfg, data, transformer, name=name, cropped=cropped)
        self.ids_funcs = {}
        self.set_ids_function("all", func_all)
        self.set_ids_function("commercial", func_commercial)
        self.set_ids_function("custom1", func_custom1)
        self.set_ids_function("vehicle", func_vehicle)
        self.set_ids_function("vehicle_all", func_vehicle_all)
        self.set_ids_function("person", func_person)
        self.set_ids_function("keypoints", func_keypoints)

        
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
            self.ids, self.map_catID = self.ids_funcs[key](self.coco)
            #if self.map_catID["id"] == "img":
            #    self.__getitem__ = self.__getitem__img
            #elif self.map_catID["id"] == "ann+img":
            #    self.__getitem__ = self.__getitem__ann_img

        else:
            raise ValueError("set ids_image_form correctly")


class coco2017_(coco_base_specific_):
    name = 'coco2017'
    def __init__(self, cfg, data='train', transformer=None, cropped=True):
        self.year = "2017"
        super(coco2017_, self).__init__(cfg, data, transformer, name="2017", cropped=cropped)

class coco2014_(coco_base_specific_):
    name = 'coco2014'
    def __init__(self, cfg, data='train', transformer=None, cropped=True):
        self.year = "2014"
        super(coco2014_, self).__init__(cfg, data, transformer, name="2014", cropped=cropped)


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
    
if __name__ == '__main__':

    print("start")
    #np.random.seed(1234)
    np.random.seed(9999)
    from easydict import EasyDict as edict
    from dataset.augmentator import get_compose_keypoints0
    from dataset.augmentator import get_compose_resize5

    cfg = edict()
    #if True:
    if False:
        year = '2014'
        cfg.PATH = '/data/public_data/COCO2014/'
        coco = coco2014_
    else:
        year = '2017'
        cfg.PATH = '/data/public_data/COCO2017/'
        coco = coco2017_

    cfg.ANNTYPE = 'bbox'
    cfg.IDS = 'all'
    cfg.BATCHSIZE = 30
    #cfg.NUM_CLASSES = 91
    cfg.NUM_CLASSES = 80
    
    # Image size for YOLO
    image_size = 416

    #fmt = "pascal_voc"
    fmt = "coco"

    
    compose_keypoints = get_compose_keypoints0(256, 192)
    compose = get_compose_resize5(image_size, image_size, fmt)
    #compose = None

    print(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.PATH = sys.argv[2]

    if sys.argv[1] == "loader":
        #cfg.FORM = "icxywh_normalized"
        
        
        check_loader(cfg, coco, compose)

    elif sys.argv[1] == "keypoints":
        cfg.FORM = "xyc"
        check_keypoints(cfg, coco, compose_keypoints)
        #check_keypoints(cfg, coco, None)
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

