
import numpy as np
from utils import operator
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset



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


def plot_NumObjects_dataset(path, data_type, object_num, content):
    """
    plot bar graph which describe number of each objects on dataset.
    """

    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as plt

    print(object_num)
    plt.figure()
    for o in object_num:
        plt.bar(range(1, len(o)+1), o,  align="center")
        print(np.argwhere(o < 1)) 
        print(np.sum(o))

    plt.legend(content)
    plt.grid()
    plt.yscale('log')
    plt.ylim([1e0, 4e5])
    plt.savefig(path + "data_"  + data_type  + ".png")


def check_loader(cfg, coco, compose=None):
    """
    """

    print("Check coco dataloader")
    #
    #
    #
    data_type = cfg.DTYPE
    objects = cfg.NUM_CLASSES
    path = cfg.LOG_SAVE

    #
    #
    #
    data_ = coco(cfg, data_type, compose)
    data_.initialize_loader()
    data_.iscrowd_exist = False

    print(data_.map_catID)

    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=cfg.CFN)

    s = 0
    object_num = np.zeros(objects)
    for batch_idx, (imgs, targets_list) in enumerate(loader):

        print(np.array(imgs).shape)
        for targets in targets_list[0]:
            print(np.array(targets).shape)
            for t in targets:
                object_num[int(t[4])] += 1
        d = (batch_idx+1)/len(loader) * 100
        s = time.time()

    print("test_loader.check_loader")
    data_type_ = data_type + "_" + cfg.IDS
    plot_NumObjects_dataset(path, data_type_, [object_num], [data_type])
    np.save(path + "data_"  + data_type_  + ".npy", object_num)



def check_bbox(cfg, coco, dir_export="./", compose=None):

    print("Check coco BBox and data augmentations")

    from utils import operator
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    print("Check coco dataloader")
    #
    #
    #
    data_type = cfg.DTYPE
    objects = cfg.NUM_CLASSES
    path = cfg.LOG_SAVE

    #
    #
    #
    data_ = coco(cfg, data_type, compose)
    data_.initialize_loader()
    data_.iscrowd_exist = False

    print(data_.map_catID)

    loader = DataLoader(data_, batch_size=cfg.BATCHSIZE,
                        shuffle=False, num_workers=2, collate_fn=cfg.CFN)


    #print(dir_export)
    path = dir_export + "bbox/"
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

