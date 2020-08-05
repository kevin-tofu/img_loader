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

    def __init__(self, cfg, transformer=None):

        self.num_data = -1
        self.batchsize = cfg.BATCHSIZE
        self.transformer = transformer
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

class base_bbox(base):

    def __init__(self, cfg, transformer=None):
        super(base_bbox, self).__init__(cfg, transformer)

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

    def augumentation_albumentations(self, img, label):
        
        #print(label, img.shape, "raw")
        # !!! caution : TODO
        label[:, 2:4] = np.clip(label[:, 2:4], 0.1, 416 - 0.1) 
        #label[:, 2:3] = np.clip(label[2:3], 0.1, img.shape[1] - label[:, 0:1] - 0.1)
        #label[:, 3:4] = np.clip(label[3:4], 0.1, img.shape[0] - label[:, 1:2] - 0.1)
        annotation = {'image': img, 'bboxes': label[:, 0:4], 'category_id': label[:, 4]}
        augmented = self.transformer(**annotation)
        img_trans = augmented['image']
        x1y1wh_trans = np.array(augmented['bboxes'])
        
        # !!! caution : TODO
        x1y1wh_trans = np.clip(x1y1wh_trans, 0, 415.9) 

        id_trans = np.array(augmented['category_id'])[:, np.newaxis]
        return img_trans, x1y1wh_trans, id_trans


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

            if self.transformer is not None:
                # augumentation by albumentations
                img_trans, x1y1wh_trans, id_trans = self.augumentation_albumentations(img, label)
            else:
                # without augumentation
                img_trans = img
                x1y1wh_trans = np.array(label[:, 0:4])
                id_trans = np.array(label[:, 4:])

            if len(x1y1wh_trans) == 0:
                continue

            ret_images.append(img_trans)
            if self.__form == "icxywh_normalized":
                b_trans = b * np.ones((x1y1wh_trans.shape[0], 1))
                xywh_trans = x1y1wh_trans / img_trans.shape[0]
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((b_trans, id_trans, xywh_trans), 1).tolist()
                ret_targets += label_trans
                
            elif self.__form == "x1y1whc":
                label_trans = np.concatenate((x1y1wh_trans, id_trans), 1).tolist()
                ret_targets.append(label_trans)

            elif self.__form == "xywhc":
                xywh_trans = np.copy(x1y1wh_trans)
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
                ret_targets.append(label_trans)

            elif self.__form == "xywhc_normalized":
                xywh_trans = x1y1wh_trans / img_trans.shape[0]
                xywh_trans[:, 0:2] += xywh_trans[:, 2:4] / 2.
                label_trans = np.concatenate((xywh_trans, id_trans), 1).tolist()
                ret_targets.append(label_trans)
            b += 1

        return ret_images, ret_targets

