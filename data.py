import os
import os.path
import numpy as np
from regex import D
import joblib

def get(batch_size=1, data_root='.', train=False, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, "data/"))
    print("Building data loader, train and test")
    ds = [] 
    if train:
        ds.append(DATASET(data_root, batch_size, True, **kwargs))
    if val:
        ds.append(DATASET(data_root, batch_size, False, **kwargs))
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class DATASET(object):
    def __init__(self, root, batch_size, train=False, input_size=224, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.train = train
        self.root = root
        self.train_picture_name_file = "./data/picture_name.txt"
        self.train_label_name_file = "./data/picture_label.txt"
        self.val_picture_name_file = "./data/picture_name.txt"
        self.val_label_name_file = "./data/picture_label.txt"
        self.picture_list = []
        self.label_list = []
        if train:
            with open(self.train_picture_name_file, "r") as picture_names:
                picture_name_list = picture_names.readlines()
                picture_list = [os.path.join(root, picture_name.strip().split(" ")[0]) for picture_name in picture_name_list]
                label_list = []
                with open(self.train_label_name_file, "r") as picture_labels:
                    label_list = [str(x).strip("\n") for x in picture_labels.readlines()]
        else:
            with open(self.val_picture_name_file, "r") as picture_names:
                picture_name_list = picture_names.readlines()
                picture_list = [os.path.join(root, picture_name.strip().split(" ")[0]) for picture_name in picture_name_list]
                label_list = []
                with open(self.train_label_name_file, "r") as picture_labels:
                    label_list = [str(x).strip("\n") for x in picture_labels.readlines()]

        # self.data_dict = joblib.load(pkl_file)
        self.batch_size = batch_size
        self.idx = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample * 1.0 / self.batch_size))
    
    @property 
    def n_sample(self):
        return len(self.data_dict["data"])
    
    def __len__(self):
        return self.n_batch 

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.idx >= self.n_batch:
            self.idx = 0
            raise StopIteration 
        else:
            img = self.data_dict['data'][self.idx * self.batch_size: (self.idx + 1) * self.batch_size].astype('float32')
            target = self.data_dict['target'][self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
            self.idx += 1
            return img, target
