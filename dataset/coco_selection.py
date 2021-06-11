



def func_all(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __ret_img = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    
    for _loop, cat in enumerate(nms):
        
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])

    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, coco.dataset['categories']
    

def func_all_pattern1(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __new_cat_list = list()

    __ret_img = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())

    #nms = [cat['name'] for cat in cats]
    nms = ["person", "car", "bus"]
    
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
        __new_cat_list.append(cat_element[0])

    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


def func_commercial(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}


    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])

    for _id in coco.getImgIds():
        id_license = coco.imgs[_id]['license']
        if id_license >= 4:
            #ret.append(cc.imgs[i]['id'])
            __ret_img.append(_id)
    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, coco.dataset['categories']
    

def func_keypoints_commercial(coco):
    __ret_ann = []
    __map_catID = {}
    __map_invcatID = {}
    
    for aid, ann in coco.anns.items():
        if ann['num_keypoints'] > 0:
            id_license = coco.imgs[ann['image_id']]['license']
            if id_license >= 4:
                __ret_ann.append([aid, ann['image_id']])

    __map_catID["id"] = "ann+img"
    return __ret_ann, __map_catID, __map_invcatID, coco.dataset['categories']


def func_custom1(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    _pickup = 200
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    #print("nms", nms, len(nms))
    for _loop, cat in enumerate(nms):
        #print(cat)
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        #print(_loop, catIds[0])
        #print(cat, len(imgIds))
        if len(imgIds) != 0:
            idx = np.random.choice(len(imgIds), _pickup)
        else:
            continue
        __ret_img += np.array(imgIds)[idx].tolist()
    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, coco.dataset['categories']

def func_vehicle(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID ={}
    __new_cat_list = list()

    #cats = coco.loadCats(coco.getCatIds())
    nms = ["truck", "car", "bus"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)

        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        __ret_img += imgIds

        cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
        __new_cat_list.append(cat_element[0])

    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


def func_vehicle_all(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __new_cat_list = list()

    #cats = coco.loadCats(coco.getCatIds())
    nms = ["truck", "car", "bus", "bicycle", "motorcycle", "airplane", "train", "boat"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
        __new_cat_list.append(cat_element[0])

        __ret_img += imgIds

    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


def func_keypoints(coco):
    __ret_ann = []
    __map_catID = {}
    __map_invcatID = {}
    for aid, ann in coco.anns.items():
        if ann['num_keypoints'] > 0:
            #__ret_img.append(ann['image_id'])
            #__ret_ann.append(aid)
            __ret_ann.append([aid, ann['image_id']])

    __map_catID["id"] = "ann+img"
    return __ret_ann, __map_catID, __map_invcatID, coco.dataset['categories']


def func_person(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __new_cat_list = list()
    nms = ["person"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
        __new_cat_list.append(cat_element[0])

        __ret_img += imgIds
    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


def func_personANDothers2(coco):
    
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __new_cat_list = list()

    __ret_img = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    _loop = 0

    for cat in nms:

        catIds = coco.getCatIds(catNms=cat)
        if cat == 'person':
            __map_catID[int(catIds[-1])] = 0
            __map_invcatID[0] = int(catIds[-1])
            _loop += 1
            cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
            __new_cat_list.append(cat_element[0])
        else:
            __map_catID[int(catIds[-1])] = _loop
            __map_invcatID[1] = 2

    #cat_element = {'supercategory': 'person', 'id': 1, 'name': 'person'}
    cat_element = {'supercategory': 'others', 'id': 2, 'name': 'others'}
    __new_cat_list.append(cat_element)

    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


def func_personANDothers2_commercial(coco):
    
    __ret_img = []
    img_ids, __map_catID, __map_invcatID, __new_cat_list = func_personANDothers2(coco)

    for _id in img_ids:
        id_license = coco.imgs[_id]['license']
        if id_license >= 4:
            __ret_img.append(_id)
    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list

def func_personANDothers(coco):
    __ret_img = []
    __map_catID = {}
    __map_invcatID = {}
    __new_cat_list = list()

    nms = ["person", "others"]
    for _loop, cat in enumerate(nms):
        catIds = coco.getCatIds(catNms=cat)
        imgIds = coco.getImgIds(catIds=catIds)
        __map_catID[int(catIds[-1])] = _loop
        __map_invcatID[_loop] = int(catIds[-1])
        cat_element = list(filter(lambda cat_loop: cat_loop['name'] == cat, coco.dataset['categories']))
        __new_cat_list.append(cat_element[0])

        __ret_img += imgIds
    __map_catID["id"] = "img"
    return __ret_img, __map_catID, __map_invcatID, __new_cat_list


