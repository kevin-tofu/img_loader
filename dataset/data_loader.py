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

        self.num_data = -1
        self.batchsize = cfg.BATCHSIZE
        self.indeces = None
        self._loop = 0
        self.normalized = True
        self.indeces = None

    def __len__(self):
        NotImplementedError()

    def initialize_dataset(self):
        NotImplementedError()
    
    def get_indeces_batches(self):

        perm = np.random.permutation(range(self.num_data))
        temp1 = list(chunks(perm, self.batchsize))
        return temp1

    def __iter__(self):
        return self

    def __next__(self):
        NotImplementedError()


class base_augmentation(base):

    def __init__(self, cfg, transformer=None):
        super(base_augmentation, self).__init__(cfg)
        self.transformer = transformer
        self.prefix = cfg.ANNTYPE#
        self.form = "icxywh_normalized"

    @property
    def prefix(self):
        return self.__prefix

    @prefix.setter
    def prefix(self, v):
        """
        setter for functions to transform data and annotations, its format.
        """
        self.__prefix = 'instances'
        if v == "bbox":
            self.__prefix = 'instances'
            self.raw_transform = self.raw_bbox
            self.augmentation_albumentations = self.augmentation_bbox
            self.format = self.format_bbox
        elif v == "pose":
            self.__prefix = 'person_keypoints'
            self.raw_transform = self.raw_keypoints
            self.augmentation_albumentations = self.augmentation_keypoints
            self.format = self.format_keypoints
        elif v == "captions":
            self.__prefix = 'captions'
            self.raw_transform = None
            self.augmentation_albumentations = None
            self.format = None

    @property
    def form(self):
        return self.__form

    @form.setter
    def form(self, v):
        if v == "icxywh_normalized":
            self.__form = v
        elif v == "x1y1whc":
            self.__form = v
        elif v == "xywhc":
            self.__form = v
        elif v == "xywhc_normalized":
            self.__form = v
        else:
            self.__form = "icxywh_normalized"
            raise ValueError("choose from [icxywh_normalized, x1y1whc, xywhc, xywhc_normalized]")

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

        if self.__form == "icxywh_normalized":
            b_trans = b * np.ones((x1y1wh_trans.shape[0], 1))
            xywh_trans = x1y1wh_trans / img_trans.shape[0]
            xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
            label_trans = np.concatenate((b_trans, id_trans, xywh_trans), 1).tolist()
            ret_targets += label_trans
        else:
            if self.__form == "x1y1whc":
                label_trans = np.concatenate((x1y1wh_trans, id_trans), 1).tolist()
            elif self.__form == "xywhc":
                xywh_trans = np.copy(x1y1wh_trans)
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
            elif self.__form == "xywhc_normalized":
                xywh_trans = x1y1wh_trans / img_trans.shape[0]
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
            ret_targets.append(label_trans)

    def format_keypoints(self, b, data_trans, ret_targets):

        keypoints, valid = data_trans[1], data_trans[2]
        ret = np.concatenate((keypoints, valid), 1)
        ret_targets.append(ret)


    def transform(self, images, targets):

        ret_images = []
        ret_targets = []

        b = 0
        #for b, (i, t) in enumerate(zip(images, targets)):
        for i, t in zip(images, targets):
            img = np.array(i)
            label = np.array(t)# (x1, y1, w, h, c) c="id for category"
            if len(t) == 0:
                continue

            if self.transformer is None:
                # without augmentation
                outputs = self.raw_transform(img, label)
            else:
                # augmentation by albumentations
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
        

