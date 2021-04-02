
import sys, os
#if __name__ == '__main__':
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import coco_loader
from dataset import test_loader
from dataset.coco_loader import coco_base_specific_


def func_cricket(coco):
    __ret_img = []
    __map_catID = {}
    __ret_img = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]

    print(nms)
    for _loop, cat in enumerate(nms):
        print(_loop, cat)
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
        print(catIds)

    __map_catID["id"] = "img"
    #print(__map_catID["id"])

    return __ret_img, __map_catID


class cocoCricket(coco_base_specific_):
    name = 'cocoCricket'
    def __init__(self, cfg, data='train', transformer=None, cropped=True):
        self.year = "2020"
        super(cocoCricket, self).__init__(cfg, data, transformer, name="_cricket1", cropped=cropped)



def test_loadcoco(cfg, coco, compose=None):

    #data_type = cfg.DTYPE
    #data_ = coco(cfg, data_type, compose)
    #func_cricket(coco.coco)
    print("coco.num_data", coco.num_data)
    print("__getitem__", coco.__getitem__(0))

    #coco._next_data()
    #coco._next_data()



if __name__ == '__main__':
    """
    """
    
    print("start")
    #np.random.seed(1234)
    #np.random.seed(9999)
    import sys
    from dataset import test_loader, data_loader
    from easydict import EasyDict as edict
    from dataset.augmentator import aug_bbox, compose_bbox3

    #compose = aug_bbox(compose_bbox3)
    compose = None

    cfg = edict()
    #if True:
    year = '2014'
    cfg.PATH = '/data/public_data/cricket/data_20210113/'
    coco = cocoCricket

    cfg.ANNTYPE = 'bbox'
    cfg.IDS = 'all'
    cfg.BATCHSIZE = 30
    cfg.NUM_CLASSES = 2
    cfg.DTYPE = 'train'
    cfg.LOG_SAVE = "./"
    cfg.CFN = data_loader.collate_fn_bbox
    cfg.iscrowd_exist = False

    
    # Image size for YOLO
    image_size = 416

    #fmt = "pascal_voc"
    fmt = "coco"

    #data_ = coco(cfg, cfg.DTYPE, compose)
    #data_.initialize_loader()
    #test_loadcoco(cfg, data_, compose=None)


    print(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.PATH = sys.argv[2]

    if sys.argv[1] == "loader":

        test_loader.check_loader(cfg, coco, compose)

    elif sys.argv[1] == "bbox":
        #cfg.FORM = "icxywh_normalized"
        cfg.FORM = "xywhc"
        
        test_loader.check_bbox(cfg, coco, "./", compose)

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

    print("end")