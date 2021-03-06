import os
import sys
import time
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

def x1y1wh_to_xywh(label):
    ret = np.copy(label)
    ret[:, 0:2] += ret[:, 2:4] / 2.
    return ret

def x1y1wh_to_xywhc_normalized(label, img):
    ret = np.copy(label) / img.shape[0]
    ret[:, 0:2] += ret[:, 2:4] / 2.
    return ret

def collate_fn_bbox(batch):
    images = [b["image"] for b in batch if len(b["bboxes"]) > 0]
    x1y1wh_trans = [b["bboxes"] for b in batch if len(b["bboxes"]) > 0]
    id_trans = [b["category_id"] for b in batch if len(b["bboxes"]) > 0]

    targets = [np.concatenate([x1y1wh_to_xywh(_x1y1wh), np.array(_id)[:, np.newaxis]], axis = 1) for _x1y1wh, _id in zip(x1y1wh_trans, id_trans)]
    img_id = [b["id_img"] for b in batch if len(b["bboxes"]) > 0]
    imsize = [b["imsize"] for b in batch if len(b["bboxes"]) > 0]
    
    return images, (targets, img_id, imsize)


def collate_fn_bbox_ixywhc(batch):

    images = [b["image"] for b in batch]
    img_id = [b["id_img"] for b in batch]
    imsize = [b["imsize"] for b in batch]
    targets = [np.concatenate([i*np.ones((len(b['bboxes']), 1)), \
                               x1y1wh_to_xywh(b['bboxes']), \
                               np.array(b['category_id'])[:, np.newaxis]], \
                               axis=1) for i, b in enumerate(batch) if len(b['bboxes']) > 0]
    
    if len(targets) > 0:
        target2 = np.concatenate(targets, axis=0)
    else:
        target2 = []

    return images, (target2, img_id, imsize)


def collate_fn_bbox_x1y1wh(batch):
    images = [b["image"] for b in batch if len(b["bboxes"]) > 0]
    x1y1wh = [b["bboxes"] for b in batch if len(b["bboxes"]) > 0]
    id_trans = [b["category_id"] for b in batch if len(b["bboxes"]) > 0]

    targets = [np.concatenate([x1y1wh_loop, np.array(_id)[:, np.newaxis]], axis = 1) for x1y1wh_loop, _id in zip(x1y1wh, id_trans)]
    img_id = [b["id_img"] for b in batch if len(b["bboxes"]) > 0]
    imsize = [b["imsize"] for b in batch if len(b["bboxes"]) > 0]
    
    return images, (targets, img_id, imsize)

def collate_fn_keypoints(batch):

    # remove data without 
    n_valid = 3
    images = [b["image"] for b in batch if len(b["keypoints"]) >= n_valid]
    targets = [b["keypoints"] for b in batch if len(b["keypoints"]) >= n_valid]
    img_id = [b["id_img"] for b in batch if len(b["keypoints"]) >= n_valid]
    imsize = [b["imsize"] for b in batch if len(b["keypoints"]) >= n_valid]
    
    return images, (targets, img_id, imsize)

def collate_fn_keypoint_image(batch):

    # remove data without 
    n_valid = 3
    images = [b["image"] for b in batch if len(b["keypoints"][0]) >= n_valid]
    targets = [b["keypoints"] for b in batch if len(b["keypoints"][0]) >= n_valid]
    img_id = [b["id_img"] for b in batch if len(b["keypoints"][0]) >= n_valid]
    center = [b["center"] for b in batch if len(b["keypoints"][0]) >= n_valid]
    scale = [b["scale"] for b in batch if len(b["keypoints"][0]) >= n_valid]
    
    return images, (targets, img_id, center, scale)


def collate_fn_images(batch):
    images = [b["image"] for b in batch if b["image"] is not None]
    imsize = [b["imsize"] for b in batch if b["image"] is not None]
    img_id = [b["id_img"] for b in batch if b["image"] is not None]

    return images, (None, img_id, imsize)

def collate_fn_images_sub(batch):
    images = [b["image"] for b in batch if b["image"] is not None]
    imsize = [b["imsize"] for b in batch if b["image"] is not None]
    img_id = [b["id_img"] for b in batch if b["image"] is not None]
    img_fname = [b["img_fname"] for b in batch if b["image"] is not None]
    #img_path = [b["img_path"] for b in batch if b["image"] is not None]
    return images, (None, img_id, imsize, img_fname)


class base(object):

    def __init__(self, cfg):

        self.batchsize = cfg.BATCHSIZE
        self.indeces = None
        self.set_keys()
        self.form = cfg.FORM
        
    @property
    def num_data(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def initialize_dataset(self):
        raise NotImplementedError()

    #
    # Erase in the future
    #
    def get_indeces_batches(self):

        perm = np.random.permutation(range(self.num_data))
        temp1 = list(chunks(perm, self.batchsize))
        return temp1

    def set_keys(self):
        self.__keys_bbox = []
        #self.__keys_bbox.append("icxywh_normalized")
        #self.__keys_bbox.append("icxywh")
        #self.__keys_bbox.append("x1y1whc")
        self.__keys_bbox.append("xywhc")
        self.__keys_bbox.append("ixywhc")
        #self.__keys_bbox.append("xywhc_normalized")
        self.__keys_keypoints = []
        self.__keys_keypoints.append("xyc")
        self.__keys_keypoints.append("xycb")

    @property
    def form(self):
        return self.__form

    @form.setter
    def form(self, v):

        print('annotation form', v)
        if self.anntype == "bbox" or self.anntype == "bbox_keyfilter":
            if v in self.__keys_bbox:
                self.__form = v
                if v == "xywhc":
                    self.collate_fn = collate_fn_bbox
                elif v == "ixywhc":
                    self.collate_fn = collate_fn_bbox_ixywhc
            else:
                raise ValueError("choose value from ", self.__keys_bbox)

        elif self.anntype == "keypoints":
            if v in self.__keys_keypoints:
                self.__form = v
                if v == "xyc":
                    self.collate_fn = collate_fn_keypoint_image
                elif v == "xycb":
                    self.collate_fn = collate_fn_keypoints
                else:
                    raise ValueError("choose value from <xyc> or <xycb>", )    
            else:
                raise ValueError("choose value from ", self.__keys_keypoints)

    @property
    def anntype(self):
        return self.__anntype

    @anntype.setter
    def anntype(self, v):
        self.__prefix = 'instances'

        if v in ["bbox", "keypoints", "bbox_keyfilter", "captions"]:
            self.__anntype = v
        else:
            raise ValueError("choose from [bbox, keypoints, bbox_keyfilter, captions]")
        
