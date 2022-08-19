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
import pandas as pd

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
# from external_evaluator import external_evaluate

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=8)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--eval_batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lamb1", type=float, default=1.)
parser.add_argument("--lamb2", type=float, default=0.03)
parser.add_argument("--data_dir", type=str, default="../../data/")
parser.add_argument("--label_file", type=str, default="../../data/")
parser.add_argument("--method", type=str, default="niqe")
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

def report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list):
    valid_pred_list = pd.Series(valid_pred_list)
    valid_gold_list = pd.Series(valid_gold_list)
    test_pred_list = pd.Series(test_pred_list)
    test_gold_list = pd.Series(test_gold_list)
    r_valid_spearman = valid_pred_list.corr(valid_gold_list, method='spearman')
    r_valid_pearson = valid_pred_list.corr(valid_gold_list, method='pearson')
    r_valid_kendall = valid_pred_list.corr(valid_gold_list, method='kendall')
    r_test_spearman = test_pred_list.corr(test_gold_list, method='spearman')
    r_test_pearson = test_pred_list.corr(test_gold_list, method='pearson')
    r_test_kendall = test_pred_list.corr(test_gold_list, method='kendall')
    return r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall

LAMB1 = args.lamb1
LAMB2 = args.lamb2
LR = args.lr

# def cal_loss_generate(gold_gaze, generate_gaze):
#     return torch.mean((gold_gaze - generate_gaze) ** 2)

def cal_loss_pair():
    return 0.0

def cal_loss_cam(cam, gold_gaze):
    return LAMB1 * torch.mean((cam - gold_gaze) ** 2)

def cal_loss_score(pred_score, gaze_score):
    return LAMB2 * ((pred_score - gaze_score) ** 2)

def run(data_dir, label_file, method, epochs, train_batch_size, eval_batch_size, train=True, device=None):

    train_dataset = MyDataset(data_dir=data_dir, label_file=label_file, split_str="train", img_size=224, device=device, gaze_dir=gaze_dir)
    valid_dataset = MyDataset(data_dir=data_dir, label_file=label_file, split_str="valid", img_size=224, device=device, gaze_dir=gaze_dir)
    test_dataset = MyDataset(data_dir=data_dir, label_file=label_file, split_str="test", img_size=224, device=device, gaze_dir=gaze_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=eval_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=True)
    valid_pred_list = []
    test_pred_list = []
    valid_gold_list = []
    test_gold_list = []
    if method == "psnr":
        pass
    elif method == "niqe":
        pass
    elif method == "piqe":
        pass
    elif method == "brisque":
        pass
    elif method == "self-imple":
        model = VisualSimilarityModel(device=device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        for epoch in range(epochs):
            print("Epoch: ", epoch, "/", epochs)
            model.train()
            tqdm_loader = tqdm.tqdm(train_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_loader:
                ###########
                # FORWARD #
                ###########
                original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                gaze_img = gaze_img.to(device)
                result = model(original_img, gaze_img, index)
                y, cam, downsample = result
                # y, cam, downsample, generate = result
                # cam is a tuple
                ########
                # LOSS #
                ########
                # piqe_result_list = []
                # niqe_result_list = []
                # brisque_result_list = []
                # for img in original_img:
                #     img_feed = img.permute(1, 2, 0).detach().cpu().numpy()
                #     piqe_result = piqe(img_feed)  
                    # Score, NoticeableArtifactsMask, NoiseMask, ActivityMask 
                    # piqe_result_list.append(piqe_result[0])
                    # niqe_result = niqe(img_feed)
                    # # Score
                    # niqe_result_list.append(niqe_result)
                    # brisque_result = brisque(img_feed) 
                    # # Score
                    # brisque_result_list.append(brisque_result)
                # piqe_result_list = torch.Tensor(piqe_result_list).to(device)
                # niqe_result_list = torch.Tensor(niqe_result_list)
                # brisque_result_list = torch.Tensor(brisque_result_list)

                loss_cam = cal_loss_cam(cam[2], gaze_img)
                loss_score = cal_loss_score(y, gaze_score.float().cuda())
                # loss_another = piqe_result_list
                # loss_pair = cal_loss_pair(y, gaze_score.float().cuda())
                # loss = (loss_cam + loss_score + loss_another).mean() # + loss_pair
                loss = (loss_cam + loss_score).mean()
                # print(loss)

                with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                    # tqdm_loader.set_description("Loss %.4f, Loss nriqa %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_another.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                    tqdm_loader.set_description("Loss %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                    f.write("Loss %.4f, Prob %s \n" % (loss.data.cpu().numpy().mean(), str(cam[1].data.cpu().numpy().mean())))
                
                ############
                # Backward #
                ############
                if train and loss.max() != 0 and loss.mean() != np.nan:
                    try:
                        optim.zero_grad()
                        loss.backward(retain_graph=True)
                        optim.step()
                    except:
                        traceback.print_exc()
                        ipdb.set_trace()
                        print("unknown error")

            model.eval()
            tqdm_eval_loader = tqdm.tqdm(valid_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                gaze_img = gaze_img.to(device)
                result = model(original_img, gaze_img, index)
                y, cam, downsample = result
                for _y_item, _gaze_score_item in zip(y, gaze_score):
                    valid_pred_list.append(_y_item)
                    valid_gold_list.append(_gaze_score_item)
            tqdm_test_loader = tqdm.tqdm(test_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_test_loader:
                original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                gaze_img = gaze_img.to(device)
                result = model(original_img, gaze_img, index)
                y, cam, downsample = result
                for _y_item, _gaze_score_item in zip(y, gaze_score):
                    test_pred_list.append(_y_item)
                    test_gold_list.append(_gaze_score_item)
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall = report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list)
            tqdm_test_loader.set_description("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
        # external_evaluate()    
        # np.save(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".npy"), next(model.parameters()).cpu().numpy())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs = {
        "epoch": args.epoch
    }    
    make_log_dir(logs)

    run(data_dir=args.data_dir, label_file=args.label_file, 
        method=args.method, epochs=args.epoch, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, device=device)
