import os
import sys
import warnings
import cv2
import ipdb
import numpy as np
import torch
import traceback
import csv
from torch.utils.data import DataLoader, Dataset
from utils import gaussian

warnings.filterwarnings("ignore")

sys.path.append("../renderer/")

import nmr_test as nmr

class MyDataset(Dataset):
    def __init__(self, data_dir, label_file, split_str, img_size, device=0, gaze_dir=''):
        '''
        static_path = "/root/autodl-tmp/experiment_picture/"
        data_dir = "/root/autodl-tmp/experiment_picture/"
        '''
        data_dir = data_dir + "experiment_picture_partial/"
        input_name_list = []
        output_name_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            # print(dirs)
            for name in files:
                # ipdb.set_trace()
                input_name_list.append(os.path.join(root, name))
                output_name_list.append(os.path.join(data_dir + root[25:], name[:-4] + ".png"))

        # print(input_name_list)
        file_dict = {}
        for file in input_name_list:
            fileprefix = "/".join(file.split("/")[:-1])
            if fileprefix not in file_dict.keys():
                file_dict[fileprefix] = []
            file_dict[fileprefix].append(file)
        # ipdb.set_trace()
        self.train_ratio = 0.6
        self.test_ratio = 0.2
        self.valid_ratio = 0.2
        self.device = device
        self.files = file_dict
        self.file_list = []
        for index_item in self.files.keys():
            self.file_list.append(index_item)

    def preprocess_csvs(self, file_list):
        answer = None
        count = 0
        for file in file_list:
            ret = self.preprocess_csv(file)
            if ret is not None:
                count += 1
                if answer is None:
                    answer = ret     
                else:
                    answer += ret
        if count > 0:  
            return answer / count
        else:
            return np.array(np.zeros((224, 224)).astype(np.uint8))

    def preprocess_csv(self, file):
        imagefile = None
        alpha = 0.5
        gaussianwh = 200
        gaussiansd = None
        dispsize = (2560, 1440)
        # ipdb.set_trace()
        with open(file, "r", encoding='gbk') as f:
            try:
                reader = csv.reader(f)
                raw = []
                for row in reader:
                    raw.append(row)

                gaze_data = []
                if len(raw) > 0:
                    if len(raw[0]) == 2:
                        gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw))
                    else:
                        gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))
                if len(gaze_data) > 0:
                    # HEATMAP
                    # Gaussian
                    gwh = gaussianwh
                    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
                    gaus = gaussian(gwh, gsdwh)
                    # matrix of zeroes
                    strt = gwh / 2
                    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
                    heatmap = np.zeros(heatmapsize, dtype=float)
                    # create heatmap
                    while gaze_data[0][0] == 0:
                        gaze_data = gaze_data[1:]
                    for i in range(1, len(gaze_data)):
                        gaze_data[i] = (gaze_data[i][0], gaze_data[i][1], (gaze_data[i][2] - gaze_data[0][2]) / 1000000)
                    gaze_data[0] = (gaze_data[0][0], gaze_data[0][1], 0)
                    # print(gaze_data)
                    for i in range(0, len(gaze_data)):
                        # get x and y coordinates
                        x = strt + gaze_data[i][0] - int(gwh / 2)
                        y = strt + gaze_data[i][1] - int(gwh / 2)
                        # correct Gaussian size if either coordinate falls outside of
                        # display boundaries
                        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
                            hadj = [0, gwh]
                            vadj = [0, gwh]
                            if 0 > x:
                                hadj[0] = abs(x)
                                x = 0
                            elif dispsize[0] < x:
                                hadj[1] = gwh - int(x - dispsize[0])
                            if 0 > y:
                                vadj[0] = abs(y)
                                y = 0
                            elif dispsize[1] < y:
                                vadj[1] = gwh - int(y - dispsize[1])
                            # add adjusted Gaussian to the current heatmap
                            try:
                                heatmap[int(y):int(y + vadj[1]), int(x):int(x + hadj[1])] += gaus[int(vadj[0]):int(vadj[1]), int(hadj[0]):int(hadj[1])] * gaze_data[i][2]
                            except:
                                # fixation was probably outside of display
                                pass
                        else:
                            # add Gaussian to the current heatmap
                            heatmap[int(y):int(y + gwh), int(x):int(x + gwh)] += gaus * gaze_data[i][2]

                    # resize heatmap
                    heatmap = heatmap[int(strt):int(dispsize[1] + strt), int(strt):int(dispsize[0] + strt)]
                    # remove zeros
                    lowbound = np.mean(heatmap[heatmap > 0])
                    heatmap[heatmap < lowbound] = lowbound
                    heatmap -= lowbound
                    heatmap = np.array(heatmap[:,560:2000]).astype(np.uint8)
                    # ipdb.set_trace()
                    heatmap = cv2.resize(heatmap, (224, 224))
                    # ipdb.set_trace()
                    return heatmap
                else:
                    return np.array(np.zeros((224, 224)).astype(np.uint8))
            except:
                traceback.print_exc()
                return np.array(np.zeros((224, 224)).astype(np.uint8))

    def open_and_process_image(self, file_img):
        splits = file_img.split("/")
        split_recombine = splits[:3] + splits[4:]
        path = "/".join(split_recombine)
        return cv2.imread(path)
    
    def cal_score(self, file_list):
        score = 0.0
        for file in file_list:
            score += eval(file.split("/")[-1].split("-")[-1].split(".")[0].strip("score"))
        return score / len(file_list)

    def __getitem__(self, index):
        file_img = self.file_list[index]
        file_list = self.files[file_img]
        gaze_img = self.preprocess_csvs(file_list)
        # gaze_img = cv2.resize(gaze_img, (224, 224))
        original_img = self.open_and_process_image(file_img)
        original_img = cv2.resize(original_img, (224, 224))
        gaze_score = self.cal_score(file_list)
        return index, gaze_img, gaze_score, original_img
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    dataset = MyDataset('/root/cgj/DualAttentionAttack/src/data/phy_attack/train', 608, 4, faces, vertices, device=0)
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    
    for img, car_box in loader:
        print(img.size(), car_box.size())
