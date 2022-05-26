import os
import sys
import warnings

import cv2
import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

sys.path.append("../renderer/")

import neural_renderer
import nmr_test as nmr


class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False, device=0, gaze_dir=''):
        self.gaze_dir = gaze_dir
        self.gazemap = []
        gazemap = os.listdir(gaze_dir)
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for i in range(len(files)):
            file = files[i]
            if distence is None:
                self.files.append(file)
                gazemap_path = os.path.join(self.gaze_dir, gazemap[i])
                gazeimg = cv2.imread(gazemap_path) 
                self.gazemap.append(gazeimg)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']

                cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
                cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
                cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

                veh_trans[0][2] = veh_trans[0][2] + 0.2

                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
                    gazemap_path = os.path.join(self.gaze_dir, gazemap[i])
                    gazeimg = cv2.imread(gazemap_path)
                    self.gazemap.append(gazeimg)
        print(len(self.files), len(self.gazemap))
        self.img_size = img_size
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures = torch.from_numpy(textures).cuda(device=device)
        self.faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=device)
        self.vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=device)
        self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size, device=device).cuda(device=device)
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        self.device = device
        # print(self.files)
    
    def set_textures(self, textures):
        self.textures = textures
    
    def __getitem__(self, index):
        # index = 5
        
        # print(index)
        # print(self.files[index])
        # ipdb.set_trace()
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
        cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
        cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

        veh_trans[0][2] = veh_trans[0][2] + 0.2

        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        
        self.mask_renderer.set_param(eye, camera_direction, camera_up)

        imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures)
        # masks = imgs_pred[:, 0, :, :] | imgs_pred[:, 1, :, :] | imgs_pred[:, 2, :, :]
        # print(masks.size())
        
        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=self.device)
        # print(img.size())
        # print(imgs_pred.size())
        imgs_pred = imgs_pred / torch.max(imgs_pred)
        
        # if self.ret_mask:
        mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
        mask = cv2.imread(mask_file)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        mask = torch.from_numpy(mask.astype('float32')).cuda(device=self.device)
        # print(mask.size())
        # print(torch.max(mask))

        total_img = img * (1-mask) + 255 * imgs_pred * mask

        gazeimg = self.gazemap[index]
        gazeimg = torch.from_numpy(gazeimg / np.max(gazeimg)).cuda(device=self.device) 

        return index, total_img.squeeze(0) , imgs_pred.squeeze(0), mask, gazeimg
        # return index, total_img.squeeze(0) , imgs_pred.squeeze(0)
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    dataset = MyDataset('../data/phy_attack/train/', 608, 4, faces, vertices, device=0)
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    
    for img, car_box in loader:
        print(img.size(), car_box.size())
