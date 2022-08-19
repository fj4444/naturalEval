import os, sys, warnings
import cv2
import torch
import tqdm
import argparse
import random
import traceback
import ipdb
import nmr_test as nmr
import numpy as np

from data_loader import MyDataset 
from model import VisualSimilarityModel
from PIL import Image
from functools import reduce
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset 
from torchvision import models 
from torchvision.transforms import Resize
from loss import niqe, piqe, brisque

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batchsize", type=int, default=2)
parser.add_argument("--lamb1", type=float, default=1.)
parser.add_argument("--lamb2", type=float, default=0.03)
parser.add_argument("--datapath", type=str, default="../../data/")
parser.add_argument("--seed", type=int, default=2333)
parser.add_argument("--device", type=int, default=0)

args = parser.parse_args()
SEED = args.seed
DEVICE = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.autograd.set_detect_anomaly(True)

log_dir = ""
gaze_dir = "./gazeimgs"

def make_log_dir(logs):
    global log_dir 
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + "-" + str(logs[key]) + "+"
    dir_name = "logs/" + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name

LAMB1 = args.lamb1
LAMB2 = args.lamb2
LR = args.lr 
BATCH_SIZE = args.batchsize 
EPOCH = args.epoch

# def cal_loss_generate(gold_gaze, generate_gaze):
#     return torch.mean((gold_gaze - generate_gaze) ** 2)

def cal_loss_pair():
    return 0.0

def cal_loss_cam(cam, gold_gaze):
    return LAMB1 * torch.mean((cam - gold_gaze) ** 2)

def cal_loss_score(pred_score, gaze_score):
    return LAMB2 * ((pred_score - gaze_score) ** 2)

def run(data_dir, epochs, train=True, batch_size=BATCH_SIZE, device=None):
    
    dataset = MyDataset(data_dir, 224, device=0, gaze_dir=gaze_dir)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
        # num_workers=2,
    )
    model = VisualSimilarityModel(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(epochs):
        print("Epoch: ", epoch, "/", epochs)
    
        tqdm_loader = tqdm.tqdm(loader)
        for cnt, (index, gaze_img, gaze_score, original_img) in enumerate(tqdm_loader):
            
            ###########
            # FORWARD #
            ###########
            # print(device)
            original_img = original_img.float().to(device).permute(0, 3, 1, 2)
            gaze_img = gaze_img.to(device)
            result = model(original_img, gaze_img, index)
            y, cam, downsample = result
            # y, cam, downsample, generate = result
            # cam is a tuple
            ########
            # LOSS #
            ########
            piqe_result_list = []
            # niqe_result_list = []
            # brisque_result_list = []
            for img in original_img:
                img_feed = img.permute(1, 2, 0).detach().cpu().numpy()
                piqe_result = piqe(img_feed)  
                # Score, NoticeableArtifactsMask, NoiseMask, ActivityMask 
                piqe_result_list.append(piqe_result[0])
                # niqe_result = niqe(img_feed)
                # # Score
                # niqe_result_list.append(niqe_result)
                # brisque_result = brisque(img_feed) 
                # # Score
                # brisque_result_list.append(brisque_result)
            piqe_result_list = torch.Tensor(piqe_result_list).to(device)
            # niqe_result_list = torch.Tensor(niqe_result_list)
            # brisque_result_list = torch.Tensor(brisque_result_list)

            loss_cam = cal_loss_cam(cam[2], gaze_img)
            loss_score = cal_loss_score(y, gaze_score.float().cuda())
            loss_another = piqe_result_list
            # loss_pair = cal_loss_pair(y, gaze_score.float().cuda())
            loss = (loss_cam + loss_score + loss_another).mean() # + loss_pair
            # print(loss)

            with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                tqdm_loader.set_description("Loss %.4f, Loss nriqa %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_another.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                f.write("Loss %.4f, Prob %s \n" % (loss.data.cpu().numpy().mean(), str(cam[1].data.cpu().numpy().mean())))
            
            ############
            # Backward #
            ############
            if train and loss.max() != 0:
                try:
                    optim.zero_grad()
                    loss.backward(retain_graph=True)
                    optim.step()
                except:
                    traceback.print_exc()
                    ipdb.set_trace()
                    print("unknown error")
            
        # np.save(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".npy"), next(model.parameters()).cpu().numpy())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs = {
        "epoch": EPOCH
    }    
    make_log_dir(logs)
    # train_dir = os.path.dir(args.datapath, 'train/')
    # test_dir = os.path.dir(args.datapath, 'test/')
    train_dir = args.datapath
    test_dir = args.datapath

    run(train_dir, epochs=EPOCH, device=device)
