
import os
import glob
from skimage import io
import numpy as np
#from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset

class imgloader(Dataset):
    def __init__(self, __dir, transformer=None):
        super(imgloader, self).__init__()
        self.__imgdir = __dir
        #self.__img_files = glob.glob(__dir + "*.jpg")
        self.__img_files = sorted(glob.glob(__dir + "*.jpg"))
        self.__ids = range(len(self.__img_files))
        self.__transformer = transformer

    @property
    def num_data(self):
        return len(self.__ids)
    
    def __len__(self):
        return self.num_data

    def get_imgname(self, img_id):
        img_path = self.__img_files[img_id]
        basename = os.path.basename(img_path)
        return basename

    def __getitem__(self, i):

        img_id = self.__ids[i]
        img_path = self.__img_files[img_id]
        basename = os.path.basename(img_path)

        #print(img_path)
        if os.path.exists(img_path) == False:
            #print("no image")
            return {"image":None, "img_path":img_path, "img_fname":basename, "imsize":None, "id_img":img_id}
        else:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))
            img_shape = img.shape
        
        if self.__transformer is not None:
            augmented = self.__transformer(image=img)
            augmented["img_path"] = img_path
            augmented["imsize"] = (img_shape[1], img_shape[0])
            augmented["id_img"] = img_id
            augmented["img_path"] = img_path
            augmented["img_fname"] = basename
        else:
            augmented = {"image":img, "img_path":img_path, "img_fname":basename, "imsize":(img_shape[1], img_shape[0]), "id_img":img_id}

        return augmented