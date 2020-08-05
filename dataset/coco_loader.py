
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from pycocotools.coco import COCO
from skimage import io

if __name__ == '__main__':
    import data_loader
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import data_loader

class coco_base(data_loader.base_bbox):

    def __init__(self, cfg, data='train', transformer = None, name="2014"):

        super(coco_base, self).__init__(cfg, transformer)
        self.n_class = cfg.NUM_CLASSES
        self.coco = None
        self.form = "icxywh_normalized"
        #self.ids_image_form = "all"
        self.ids_image_form = cfg.IDS

        self.pycocoloader(cfg, data, transformer, name)

    def pycocoloader(self, cfg, data, transformer, name):

        self.data_dir = cfg.PATH
        self.exception_ids = ['']
        self.dataType = self.get_datatype(data, name) # train2014
        self.prefix = cfg.ANNTYPE
        self.img_dir = self.data_dir + '/images/' + self.dataType + '/'
        self.annfname = '%s/annotations/%s_%s.json'%(self.data_dir, self.prefix, self.dataType)

        print(self.annfname)
        self.coco = COCO(self.annfname)
        #self.ids_img = self.coco.getImgIds()
        self.ids_img = self.get_ids_image()

        if data == 'train' or data == 'val' or data == 'test':
            self.num_data = len(self.ids_img)
        elif data == 'check':
            self.num_data = 500

        self.indeces_batchs = self.get_indeces_batches()

    def get_datatype(self, data, name):
        if data == 'check':
            return 'val' + name
        else:
            return data + name

    @property
    def prefix(self):
        return self.__prefix

    @prefix.setter
    def prefix(self, v):
        self.__prefix = 'instances'
        if v == "bbox":
            self.__prefix = 'instances'
            self. __next__ = self.next_bbox
        elif v == "pose":
            self.__prefix = 'person_keypoints'
            self. __next__ = self.next_bbox
        elif v == "captions":
            self.__prefix = 'captions'
            self. __next__ = self.next_bbox
        
    @property
    def ids_image_form(self):
        return self.__ids_image

    @ids_image_form.setter
    def ids_image_form(self, v):
        if v == "all":
            self.__ids_image_form = v
        elif v == "commercial":
            self.__ids_image_form = v
        else:
            self.__ids_image_form = "commercial"
            raise ValueError("choose from [all, commercial]")
    
    def get_ids_image(self):

        if self.__ids_image_form == "all":
            ret = self.coco.getImgIds()
        elif self.__ids_image_form == "commercial":
            #ret = self.coco.getImgIds()
            ret = []
            for _id in self.coco.getImgIds():
                id_license = self.coco.imgs[_id]['license']
                if id_license >= 4:
                    #ret.append(cc.imgs[i]['id'])
                    ret.append(_id)
        return ret

    def initialize_dataset(self):

        self.get_ids_image()
        self.indeces_batchs = self.get_indeces_batches()

    def next_bbox(self):
        if self._loop >= len(self.indeces_batchs):
            self._loop = 0
            self.initialize_dataset()
            raise StopIteration()

        _ids = self.indeces_batchs[self._loop]
        img_list, target_list = self.load_bbox(_ids)
        self._loop += 1
        return [img_list, target_list]

    #def __next__(self):
    #    return self.next_bbox()
    

    def __len__(self):#length of mini-batches
        return self.num_data // self.batchsize

    def categories(self):
        nms = [str(i+1) for i in range(self.n_class)]
        return nms

    def load_bbox(self, _ids):
        #https://pytorch.org/docs/stable/_modules/torchvision/datasets/coco.html#CocoDetection
        img_list = []
        target_list = []
        for i, _v in enumerate(_ids):
            
            #img
            img_id = self.ids_img[_v]
            img_name = self.coco.imgs[img_id]['file_name']
            img_path = self.img_dir + img_name
            if os.path.exists(img_path) == False:
                print("no file")
                continue
            else:
                img = io.imread(img_path)
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

            for ann in anns:
                #print(ann)
                if len(ann['bbox']) == 0:
                    print("NO BOX")
                    continue

                if ann['category_id'] >= self.n_class:
                    print('category_id : ', ann['category_id'])
                    continue

                x1 = float(ann['bbox'][0])
                y1 = float(ann['bbox'][1])
                w = float(ann['bbox'][2])
                h = float(ann['bbox'][3])
                target.append([x1, y1, w, h, ann['category_id']])

            if len(target) == 0:
                continue

            target_list.append(target)
            img_list.append(img)
        
        img_list, target_list = self.transform(img_list, target_list)

        return img_list, target_list


class coco2014(coco_base):
    name = 'coco2014'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
        super(coco2014, self).__init__(cfg, data, transformer, name="2014")

class coco2017(coco_base):
    name = 'coco2017'
    use = 'localization'
    def __init__(self, cfg, data='train', transformer=None):
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


def test_bbox(cfg, coco, compose=None):

    print("Check coco BBox and data augmentations")

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl


    data = coco(cfg, 'check', compose)
    data.initialize_dataset()
    data.__form = "x1y1whc"

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
            c_box = draw_box(c, t, data.__form)
            pl.clf()
            pl.imshow(c_box)
            pl.savefig(fname)



def test_loader(cfg, coco, compose=None):

    print("Check coco dataloader")

    data_ = coco(cfg, 'val', compose)
    data_.initialize_dataset()
    data_.__form = "icxywh_normalized"
    #print(data_train.coco.anns.keys())

    for batch_idx, (img, target) in enumerate(data_):
        print(np.array(img).shape, np.array(target).shape)
        #print(target)
        d = batch_idx/len(data_) * 100
        print('[{} / {}({:.1f}%)]'.__format(batch_idx, len(data_), d))


def test_licence(cfg, coco, compose):

    print("Check Licenses on images")

    for dtype in ["train", "val"]:
        data_ = coco(cfg, dtype, None)
        data_.initialize_dataset()
        data_.__form = "icxywh_normalized"
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


"""
#segmentation, num_keypoints, area, iscrowd, 
# keypoints, image_id, bbox, category_id, id
#print(cc.anns[_id])
#print(cc.anns[_id]['keypoints'])#row, col, num
#print(len(cc.anns[_id]['keypoints']))#51
#print(cc.anns[_id]['num_keypoints'])
"""
def test_annotations(cfg, coco, compose):
    
    print("Check Annotations on images")

    for dtype in ["train", "val"]:

        fname = cfg.PATH + "annotations/person_keypoints_" + dtype + "2017.json"
        #fname = cfg.PATH + "annotations/person_keypoints_" + dtype + "2014.json"
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


if __name__ == '__main__':

    print("start")
    np.random.seed(1234)
    from easydict import EasyDict as edict
    from dataset.augmentator import get_compose

    cfg = edict()
    #if True:
    if False:
        cfg.PATH = '/data/public_data/COCO2014/'
        coco = coco2014
    else:
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
    #compose = None

    test_bbox(cfg, coco, compose)
    #test_loader(cfg, coco, compose)
    #test_licence(cfg, coco, compose)    
    #test_annotations(cfg, coco, compose)


    print("end")





