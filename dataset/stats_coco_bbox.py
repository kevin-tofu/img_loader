
import argparse
from pycocotools.coco import COCO
import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from coco_selection import func_all

def main(args):

    fname = args.coco_directory + 'annotations/' + args.coco_file
    cocotool = COCO(fname)

    #stats_images(args, cocotool)
    stats_bbox(args, cocotool)



def stats_images(args, cocotool):
    
    ids = cocotool.getImgIds()
    print("ids", ids)

    ann_imgs = cocotool.loadImgs([ids[0]])
    #img_name = cocotool.loadImgs(ids[np.random.randint(0,len(ids))])[0]
    #img_name = cocotool.imgs[ids]['file_name']
    ann_img = ann_imgs[0]
    #img_name = cocotool.loadImgs([ids[0]])[0]
    
    print(ann_imgs)
    print(ann_img, ann_img["path"], ann_img["file_name"])
    print(ann_img["width"], ann_img["height"])
    #img = io.imread(args.fname + 'images/' + ann_img["fname"])
    #print(img.shape)


def stats_bbox(args, cocotool):

    
    __ret_img, __map_catID, __map_invcatID, cats = func_all(cocotool)
    nms=[cat['name'] for cat in cats]

    print('cat_name:', nms, 'len(nms):', len(nms))

    #ann_ids = cocotool.getAnnIds(imgIds=img_id, iscrowd=False)
    img_id = cocotool.getImgIds()
    ann_ids = cocotool.getAnnIds(imgIds=img_id)
    anns = cocotool.loadAnns(ann_ids)

    bbox_list = list()
    cat_list = list()
    for ann in anns:
        #print(ann)
        #print(ann['bbox'])
        bbox_list.append(ann['bbox'])
        cat_list.append(__map_catID[ann['category_id']])

    bbox_list = np.array(bbox_list)
    #kmeans = sklearn.cluster.KMeans(n_clusters=9, init='k-means++', n_init=20, max_iter=300,)
    #kmeans.fit_predict(bbox_list)
    model = KMeans(n_clusters=9, random_state=100).fit(bbox_list[:, 2:4]) 

    print("cluster_centers_")
    print(model.cluster_centers_)

    scale_1 = np.array([1. / 1920, 1. / 1080])
    scale_416 = np.array([416 / 1920, 416 / 1080])

    center_1 = model.cluster_centers_ * scale_1
    center_416 = model.cluster_centers_ * scale_416
    area_416 = center_416[:, 0] * center_416[:, 1]

    area_416_argsort = argsort_array = np.argsort(area_416)
    center_416_sort = center_416[area_416_argsort]
    print("cluster_centers_")
    print(center_1)
    print(center_416)
    print(area_416)

    print("center_416_sort", center_416_sort)

    print("center_416_sort_stride : ", 8)
    print(center_416_sort[0:3] / 8)
    print("center_416_sort_stride : ", 16)
    print(center_416_sort[3:6] / 16)
    print("center_416_sort_stride : ", 32)
    print(center_416_sort[6:9] / 32)

    import collections, math
    c = collections.Counter(cat_list)
    #print(c)

    count = [c[i] for i in range(80)]
    print('count', count, len(count))
    
    count_ratio = list()
    for i, c in enumerate(count):
        value = c
        sum_others = 0
        if i > 0:
            sum_others += np.sum(np.array(count)[0:i])
        if i != len(count) - 1:
            sum_others += np.sum(np.array(count)[(i+1)::])

        print('value / sum_others : ', value, '/', sum_others)
        #temp = float(value) / float(sum_others)
        temp = float(sum_others) / float(value)
        count_ratio.append(round(temp, 1))

    #count_ratio = np.array(count_ratio)
    print('count_ratio:', count_ratio, len(count_ratio))


if __name__ == '__main__':

    __dir_default = '/data/public_data/COCO2017/'
    __file_default = 'instances_train2017.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_directory', '-S', type=str, default=__dir_default, help='')
    parser.add_argument('--coco_file', '-J', type=str, default=__file_default, help='')
    args = parser.parse_args()

    main(args)