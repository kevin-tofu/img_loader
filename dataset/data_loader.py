import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

def chunks2(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in l:
		yield l[i:i + n]

class base(object):

    def __init__(self, cfg):

        self.batchsize = cfg.BATCHSIZE
        self.indeces = None
        self.set_keys()
        
    @property
    def num_data(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def initialize_dataset(self):
        raise NotImplementedError()
    
    def get_indeces_batches(self):

        perm = np.random.permutation(range(self.num_data))
        temp1 = list(chunks(perm, self.batchsize))
        return temp1

    def __iter__(self):
        return self

    def __next__(self):
        NotImplementedError()

    def set_keys(self):
        self.__keys_bbox = []
        self.__keys_bbox.append("icxywh_normalized")
        self.__keys_bbox.append("icxywh")
        self.__keys_bbox.append("x1y1whc")
        self.__keys_bbox.append("xywhc")
        self.__keys_bbox.append("xywhc_normalized")

        self.__keys_keypoints = []
        self.__keys_keypoints.append("xyc")

    @property
    def form(self):
        return self.__form

    @form.setter
    def form(self, v):
        if self.anntype == "bbox":
            if v in self.__keys_bbox:
                self.__form = v
            else:
                self.__form = "icxywh_normalized"
                cmt = ""
                for loop in self.__keys_bbox:
                    cmt += loop + ", "
                raise ValueError("choose from " + cmt)

        elif self.anntype == "keypoints":
            if v in self.__keys_keypoints:
                self.__form = v
            else:
                cmt = ""
                for loop in self.__keys_keypoints:
                    cmt += loop + ", "
                raise ValueError("choose from " + cmt)

    @property
    def anntype(self):
        return self.__anntype

    @anntype.setter
    def anntype(self, v):
        self.__prefix = 'instances'
        if v == "bbox":
            self.raw_transform = self.raw_bbox
            self.augmentation_albumentations = self.augmentation_bbox
            self.format = self.format_bbox
        elif v == "keypoints":
            self.raw_transform = self.raw_keypoints
            self.augmentation_albumentations = self.augmentation_keypoints
            self.format = self.format_keypoints
        elif v == "captions":
            self.raw_transform = None
            self.augmentation_albumentations = None
            self.format = None        
        else:
            raise ValueError("choose from [bbox, keypoints, captions]")
        
        self.__anntype = v

class base_augmentation(base):
    """
    The class that is image loader with doing data augmentation using albumentation.
    annotation basically is based on COCO format.

    Args:
        cfg: configuration given by EasyDict.
             cfg.ANNTYPE, cfg.Form and cfg.BATCHSIZE should be given. 
             ANNTYPE should be selected from "bbox", "pose", "captions". 
             ANNTYPE will be passed to prefix.

        transformer: Compose object from albumentations should be given.
                     image and its annotation will be augmentated by Compose.
    Example:
        cfg = edict()
        cfg.ANNTYPE = 'bbox'
        cfg.FORM = 'icxywh_normalized'
        cfg.BATCHSIZE = 32
        cm = Compose([Resize(image_height, image_width, p=1.0)],\
                      bbox_params={'format':format, 'label_fields':['category_id']})
        dataloader = base_augmentation(cfg, cm)
        imgs, annotations, dataloader.__next__()
    """
    def __init__(self, cfg, transformer=None):
        super(base_augmentation, self).__init__(cfg)
        self.transformer = transformer
        self.anntype = cfg.ANNTYPE
        self.form = cfg.FORM #"icxywh_normalized"


    def raw_bbox(self, img, label):
        return (img, np.array(label[:, 0:4]), np.array(label[:, 4:]))

    def augmentation_bbox(self, img, label):
        
        #print(label, img.shape, "raw")
        # !!! caution : TODO
        label[:, 2:4] = np.clip(label[:, 2:4], 0.1, 416 - 0.1) 
        annotation = {'image': img, 'bboxes': label[:, 0:4], 'category_id': label[:, 4]}

        augmented = self.transformer(**annotation)
        img_trans = augmented['image']

        x1y1wh_trans = np.array(augmented['bboxes'])
        # !!! caution : TODO
        x1y1wh_trans = np.clip(x1y1wh_trans, 0, 415.9) 
        id_trans = np.array(augmented['category_id'])[:, np.newaxis]

        if len(x1y1wh_trans) == 0:
            return None
        return (img_trans, x1y1wh_trans, id_trans)

    def raw_keypoints(self, img, keypoints):
        #print(img.shape)
        if keypoints is list:
            keypoints = np.array(keypoints)
        keypoints_all = keypoints.reshape((keypoints.shape[0] * keypoints.shape[1], 3))

        _class = []
        _person = []
        for loop in range(keypoints.shape[0]):
            _class.append(range(keypoints.shape[1]))
            _person.append(np.ones(keypoints.shape[1]).tolist())
        _class = np.array(_class).reshape((keypoints.shape[0] * keypoints.shape[1], 1))
        _person = np.array(_person).reshape((keypoints.shape[0] * keypoints.shape[1], 1))
        cp = np.concatenate((_class, _person), 1)

        return (img, keypoints_all[:, 0:2], cp)

    def augmentation_keypoints(self, img, keypoints):

        #keypoints (1, 17, 3)
        keypoints = np.array(keypoints)
        _, _, cp = self.raw_keypoints(img, keypoints)
        _class, _person = cp[:, 0], cp[:, 1]

        # !!! caution : TODO
        keypoints_all = keypoints.reshape((keypoints.shape[0] * keypoints.shape[1], 3))
        keypoints_all[:, 0:2] = np.clip(keypoints_all[:, 0:2], 0.1, 416 - 0.1)
        conf = np.argwhere((keypoints_all[:, 2] == 2))[:, 0]
        #person_select = [str(l) for l in person[conf]]

        class_select = _class[conf]
        person_select = _person[conf]

        annotation = {'image': img, \
                      'keypoints':keypoints_all[:, 0:2][conf], \
                      'class': class_select, \
                      'person': person_select}

        augmented = self.transformer(**annotation)
        img_trans = augmented['image']
        key_trans = np.array(augmented['keypoints'])
        class_trans = np.array(augmented['class'])[:, np.newaxis]
        person_trans = np.array(augmented['person'])[:, np.newaxis]
        
        # !!! caution : TODO
        #keypoints_trans = np.clip(keypoints_trans, 0, 415.9) 
        if len(key_trans) == 0:
            return None
                
        return (img_trans, key_trans, np.concatenate((class_trans, person_trans), 1))

    def format_bbox(self, b, data_trans, ret_targets):

        img_trans, x1y1wh_trans, id_trans = data_trans

        if self.form == "icxywh_normalized":
            b_trans = b * np.ones((x1y1wh_trans.shape[0], 1))
            xywh_trans = x1y1wh_trans / img_trans.shape[0]
            xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
            label_trans = np.concatenate((b_trans, id_trans, xywh_trans), 1).tolist()
            ret_targets += label_trans
        elif self.form == "icxywh":
            b_trans = b * np.ones((x1y1wh_trans.shape[0], 1))
            xywh_trans = x1y1wh_trans / 1.
            xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
            label_trans = np.concatenate((b_trans, id_trans, xywh_trans), 1).tolist()
            ret_targets += label_trans
        else:
            if self.form == "x1y1whc":
                label_trans = np.concatenate((x1y1wh_trans, id_trans), 1).tolist()
            elif self.form == "xywhc":
                xywh_trans = np.copy(x1y1wh_trans)
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
            elif self.form == "xywhc_normalized":
                xywh_trans = x1y1wh_trans / img_trans.shape[0]
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
            ret_targets.append(label_trans)

    def format_keypoints(self, b, data_trans, ret_targets):

        keypoints, valid = data_trans[1], data_trans[2]
        ret = np.concatenate((keypoints, valid), 1)
        ret_targets.append(ret)

    def transform(self, images, targets):
        #
        ret_images = []
        ret_targets = []
        b = 0
        #for b, (i, t) in enumerate(zip(images, targets)):
        for i, t in zip(images, targets):
            img = np.array(i)
            label = np.array(t)#
            if len(t) == 0:
                continue

            if self.transformer is None:
                # without augmentation
                outputs = self.raw_transform(img, label)
            else:
                # Do augmentation by albumentations
                outputs = self.augmentation_albumentations(img, label)

            if outputs is None:
                continue
            
            #print(outputs[0].shape)
            #ret_images.append(outputs[0])
            ret_images.append(outputs[0].tolist())
            self.format(b, outputs, ret_targets)
            
            b += 1

        # the size of images are ALL different. so you cannot make it np.array.
        # without resizing.
        return ret_images, ret_targets
        

def x1y1wh_to_xywh(label):
    ret = np.copy(label)
    ret[:, 0:2] += label[:, 2:4] / 2.
    return ret
def x1y1wh_to_xywhc_normalized(label, img):
    ret = np.copy(label) / img.shape[0]
    ret[:, 0:2] += label[:, 2:4] / 2.
    return ret

class base_augmentation0(base):
    """
    The class that is image loader with doing data augmentation using albumentation.
    annotation basically is based on COCO format.

    Args:
        cfg: configuration given by EasyDict.
             cfg.ANNTYPE, cfg.Form and cfg.BATCHSIZE should be given. 
             ANNTYPE should be selected from "bbox", "pose", "captions". 
             ANNTYPE will be passed to prefix.

        transformer: Compose object from albumentations should be given.
                     image and its annotation will be augmentated by Compose.
    Example:
        cfg = edict()
        cfg.ANNTYPE = 'bbox'
        cfg.FORM = 'icxywh_normalized'
        cfg.BATCHSIZE = 32
        cm = Compose([Resize(image_height, image_width, p=1.0)],\
                      bbox_params={'format':format, 'label_fields':['category_id']})
        dataloader = base_augmentation(cfg, cm)
        imgs, annotations, dataloader.__next__()
    """
    def __init__(self, cfg, transformer=None):
        super(base_augmentation0, self).__init__(cfg)
        self.transformer = transformer
        self.anntype = cfg.ANNTYPE
        self.form = cfg.FORM #"icxywh_normalized"


    def raw_bbox(self, images, labels):

        x1y1wh_trans = []
        id_trans = []
        for i, l in enumerate(labels):
            labels[i] = np.array(l)
            labels[i][:, 2:4] = np.clip(labels[i][:, 2:4], 0.1, 416 - 0.1) 
            #print(labels[i])
            x1y1wh_trans.append(labels[i][0:4])
            id_trans.append(labels[i][4:])

        return (images, x1y1wh_trans, id_trans)

    def augmentation_bbox(self, images, labels):
        
        #print(label, img.shape, "raw")
        # !!! caution : TODO
        for i, l in enumerate(labels):
            labels[i] = np.array(l)
            labels[i][:, 2:4] = np.clip(labels[i][:, 2:4], 0.1, 416 - 0.1)
            #x1y1wh_to_xywh()

        augmented = [self.transformer(image = img, bboxes=label[:, 0:4], category_id= label[:, 4]) for img, label in zip(images, labels)]
        

        #img_trans = augmented['image']
        img_trans = [a["image"] for a in augmented if len(a["bboxes"]) > 0]
        x1y1wh_trans = [np.clip(a["bboxes"], 0, 415.9)  for a in augmented if len(a["bboxes"]) > 0]
        id_trans = [np.array(a['category_id'])[:, np.newaxis]  for a in augmented if len(a["bboxes"]) > 0]

        
        #no_bbox = []
        #for i, a in enumerate(augmented):
        #    if len(a["bboxes"]) == 0:
        #        no_bbox.append(i)
        #if len(no_bbox) > 0:
        #    print("no_bbox", len(no_bbox))
        #    dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
        #    img_trans = dellist(img_trans, no_bbox)
        #    x1y1wh_trans = dellist(x1y1wh_trans, no_bbox)
        #    id_trans = dellist(id_trans, no_bbox)

        return (img_trans, x1y1wh_trans, id_trans)

    def raw_keypoints(self, img, keypoints):
        #print(img.shape)
        if keypoints is list:
            keypoints = np.array(keypoints)
        keypoints_all = keypoints.reshape((keypoints.shape[0] * keypoints.shape[1], 3))

        _class = []
        _person = []
        for loop in range(keypoints.shape[0]):
            _class.append(range(keypoints.shape[1]))
            _person.append(np.ones(keypoints.shape[1]).tolist())
        _class = np.array(_class).reshape((keypoints.shape[0] * keypoints.shape[1], 1))
        _person = np.array(_person).reshape((keypoints.shape[0] * keypoints.shape[1], 1))
        cp = np.concatenate((_class, _person), 1)

        return (img, keypoints_all[:, 0:2], cp)

    def augmentation_keypoints(self, img, keypoints):

        #keypoints (1, 17, 3)
        keypoints = np.array(keypoints)
        _, _, cp = self.raw_keypoints(img, keypoints)
        _class, _person = cp[:, 0], cp[:, 1]

        # !!! caution : TODO
        keypoints_all = keypoints.reshape((keypoints.shape[0] * keypoints.shape[1], 3))
        keypoints_all[:, 0:2] = np.clip(keypoints_all[:, 0:2], 0.1, 416 - 0.1)
        conf = np.argwhere((keypoints_all[:, 2] == 2))[:, 0]
        #person_select = [str(l) for l in person[conf]]

        class_select = _class[conf]
        person_select = _person[conf]

        annotation = {'image': img, \
                      'keypoints':keypoints_all[:, 0:2][conf], \
                      'class': class_select, \
                      'person': person_select}

        augmented = self.transformer(**annotation)
        img_trans = augmented['image']
        key_trans = np.array(augmented['keypoints'])
        class_trans = np.array(augmented['class'])[:, np.newaxis]
        person_trans = np.array(augmented['person'])[:, np.newaxis]
        
        # !!! caution : TODO
        #keypoints_trans = np.clip(keypoints_trans, 0, 415.9) 
        if len(key_trans) == 0:
            return None
                
        return (img_trans, key_trans, np.concatenate((class_trans, person_trans), 1))

    def format_bbox(self, x1y1wh_trans, id_trans, images):

        if self.form == "icxywh_normalized":
            pass
        elif self.form == "icxywh":
            pass
        else:
            if self.form == "x1y1whc":
                ret = [np.concatenate([_x1y1wh, _id], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]

            elif self.form == "xywhc":
                #ret = [np.concatenate([_x1y1wh, _id], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]
                ret = [np.concatenate([x1y1wh_to_xywh(_x1y1wh), _id], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]

            elif self.form == "xywhc_normalized":
                ret = [np.concatenate([x1y1wh_to_xywhc_normalized(_x1y1wh, img), _id], axis = 1) for _x1y1wh, _id, img in zip(x1y1wh_trans, id_trans, images)]
        
        return ret


    def format_keypoints(self, b, data_trans, ret_targets):

        keypoints, valid = data_trans[1], data_trans[2]
        ret = np.concatenate((keypoints, valid), 1)
        ret_targets.append(ret)

    def transform(self, images, targets):
        #
        if self.transformer is None:
            # without augmentation
            img_trans, x1y1wh_trans, id_trans = self.raw_transform(images, targets)
        else:
            # Do augmentation by albumentations
            img_trans, x1y1wh_trans, id_trans = self.augmentation_albumentations(images, targets)

        ret_targets = self.format(x1y1wh_trans, id_trans, img_trans)

        return img_trans, ret_targets
        


class base_augmentation_3d(base_augmentation):
    """
    The class that is image loader with doing data augmentation using albumentation.
    annotation basically is based on COCO format.

    Args:
        cfg: configuration given by EasyDict.
             cfg.ANNTYPE, cfg.Form and cfg.BATCHSIZE should be given. 
             ANNTYPE should be selected from "bbox", "pose", "captions". 
             ANNTYPE will be passed to prefix.

        transformer: Compose object from albumentations should be given.
                     image and its annotation will be augmentated by Compose.
    Example:
        cfg = edict()
        cfg.ANNTYPE = 'bbox'
        cfg.FORM = 'icxywh_normalized'
        cfg.BATCHSIZE = 32
        cm = Compose([Resize(image_height, image_width, p=1.0)],\
                      bbox_params={'format':format, 'label_fields':['category_id']})
        dataloader = base_augmentation(cfg, cm)
        imgs, annotations, dataloader.__next__()
    """
    def __init__(self, cfg, transformer=None):
        super(base_augmentation_3d, self).__init__(cfg)
        self.transformer = transformer
        self.anntype = cfg.ANNTYPE
        self.form = cfg.FORM #"icxywh_normalized"
        