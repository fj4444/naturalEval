import os, sys, warnings
import cv2
import torch
import tqdm
import argparse
import random
# supervisor
import traceback
import ipdb
import inspect
from gpu_mem_track import MemTracker # 引用显存跟踪代码

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
from niqe import niqe
from piqe import piqe
from brisque import brisque
from joblib import load
# from external_evaluator import external_evaluate

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=8)
parser.add_argument("--train_batch_size", type=int, default=6)
parser.add_argument("--eval_batch_size", type=int, default=6)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lamb1", type=float, default=1.)
parser.add_argument("--lamb2", type=float, default=0.03)
parser.add_argument("--data_dir", type=str, default="../../data/")
parser.add_argument("--label_file", type=str, default="../../data/")
parser.add_argument("--method", type=str, default="niqe")
parser.add_argument("--train", default=False, action='store_true')
parser.add_argument("--eval", default=False, action='store_true')
parser.add_argument("--test", default=False, action='store_true')
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
    valid_pred_list_processed = pd.Series([float(x[0]) for x in valid_pred_list])
    valid_gold_list_processed = pd.Series(valid_gold_list)
    test_pred_list_processed = pd.Series([float(x[0]) for x in test_pred_list])
    test_gold_list_processed = pd.Series(test_gold_list)
    r_valid_spearman = valid_pred_list_processed.corr(valid_gold_list_processed, method='spearman')
    r_valid_pearson = valid_pred_list_processed.corr(valid_gold_list_processed, method='pearson')
    r_valid_kendall = valid_pred_list_processed.corr(valid_gold_list_processed, method='kendall')
    r_test_spearman = test_pred_list_processed.corr(test_gold_list_processed, method='spearman')
    r_test_pearson = test_pred_list_processed.corr(test_gold_list_processed, method='pearson')
    r_test_kendall = test_pred_list_processed.corr(test_gold_list_processed, method='kendall')
    print(r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
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

def run(data_dir, label_file, method, epochs, train_batch_size, eval_batch_size, use_train=True, use_eval=True, use_test=True, device=None):

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
    gpu_tracker.track()
    if method == "psnr":
        pass
    elif method == "niqe":
        gpu_tracker.track()
        if use_eval:
            tqdm_eval_loader = tqdm.tqdm(valid_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                for img, gold_score in zip(gaze_img, gaze_score):
                    niqe_score = niqe(img)
                    valid_pred_list.append(niqe_score)
                    valid_gold_list.append(gold_score)
        if use_test:
            tqdm_test_loader = tqdm.tqdm(test_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                for img, gold_score in zip(gaze_img, gaze_score):
                    niqe_score = niqe(img)
                    test_pred_list.append(niqe_score)
                    test_gold_list.append(gold_score)
        if use_eval and use_test:
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall = report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list)
            tqdm_test_loader.set_description("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
            with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
                f.write("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
    elif method == "piqe":
        gpu_tracker.track()
        if use_eval:
            tqdm_eval_loader = tqdm.tqdm(valid_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                for img, gold_score in zip(gaze_img, gaze_score):
                    piqe_score = piqe(img)
                    valid_pred_list.append(piqe_score[0])
                    valid_gold_list.append(gold_score)
        if use_test:
            tqdm_test_loader = tqdm.tqdm(test_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                for img, gold_score in zip(gaze_img, gaze_score):
                    piqe_score = piqe(img)
                    test_pred_list.append(piqe_score[0])
                    test_gold_list.append(gold_score)
        if use_eval and use_test:
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall = report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list)
            tqdm_test_loader.set_description("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
            with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
                f.write("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
    elif method == "brisque":
        gpu_tracker.track()
        if use_eval:
            tqdm_eval_loader = tqdm.tqdm(valid_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                for gold_score, img in zip(gaze_score, gaze_img):
                    feature = brisque(img)
                    feature = feature.reshape(1, -1)
                    clf = load('svr_brisque.joblib')
                    pred_score = clf.predict(feature)[0]
                    valid_pred_list.append(pred_score)
                    valid_gold_list.append(gold_score)
        if use_test:
            tqdm_test_loader = tqdm.tqdm(test_loader)
            for (index, gaze_img, gaze_score, original_img) in tqdm_test_loader:
                for gold_score, img in zip(gaze_score, gaze_img):
                    feature = brisque(img)
                    feature = feature.reshape(1, -1)
                    clf = load('svr_brisque.joblib')
                    pred_score = clf.predict(feature)[0]
                    test_pred_list.append(pred_score)
                    test_gold_list.append(gold_score)
        if use_eval and use_test:
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall = report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list)
            tqdm_test_loader.set_description("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
            with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
                f.write("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
            r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
    elif method == "self-imple":
        model = VisualSimilarityModel(device=device)
        gpu_tracker.track()
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        for epoch in range(epochs):
            gpu_tracker.track()
            torch.cuda.empty_cache()
            print("Epoch: ", epoch, "/", epochs)
            if use_train:
                model.train()
                tqdm_loader = tqdm.tqdm(train_loader)
                for (index, gaze_img, gaze_score, original_img) in tqdm_loader:
                    gpu_tracker.track()
                    ###########
                    # FORWARD #
                    ###########
                    torch.cuda.empty_cache()
                    gpu_tracker.track()
                    original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                    gaze_img = gaze_img.to(device)
                    result = model(original_img, gaze_img, index)
                    y, cam, downsample = result
                    gpu_tracker.track()
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
                    gpu_tracker.track()
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                        # tqdm_loader.set_description("Loss %.4f, Loss nriqa %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_another.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                        tqdm_loader.set_description("Loss %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                        f.write("Loss %.4f, Prob %s \n" % (loss.data.cpu().numpy().mean(), str(cam[1].data.cpu().numpy().mean())))
                    
                    ############
                    # Backward #
                    ############
                    if loss.max() != 0 and loss.mean() != np.nan:
                        try:
                            optim.zero_grad()
                            loss.backward(retain_graph=True)
                            optim.step()
                        except:
                            traceback.print_exc()
                            ipdb.set_trace()
                            print("unknown error")
                gpu_tracker.track()
                # debug to save checkpoint
                # ipdb.set_trace()
                torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint_" + str(epoch) + ".pth"))
                np.save(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".npy"), next(model.parameters()).cpu().detach().numpy())
            if use_eval:    
                model.eval()
                if not use_train:
                    model.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".pth")))
                tqdm_eval_loader = tqdm.tqdm(valid_loader)
                for (index, gaze_img, gaze_score, original_img) in tqdm_eval_loader:
                    original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                    gaze_img = gaze_img.to(device)
                    result = model(original_img, gaze_img, index)
                    y, cam, downsample = result
                    gpu_tracker.track()
                    for _y_item, _gaze_score_item in zip(y.cpu().detach().numpy(), gaze_score.cpu().detach().numpy()):
                        valid_pred_list.append(_y_item)
                        valid_gold_list.append(_gaze_score_item)
                    gpu_tracker.track()
                    loss_cam = cal_loss_cam(cam[2], gaze_img)
                    loss_score = cal_loss_score(y, gaze_score.float().cuda())
                    loss = (loss_cam + loss_score).mean()
                    tqdm_eval_loader.set_description("Loss %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
            if use_test:
                model.eval()
                if not use_train:
                    model.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".pth")))
                tqdm_test_loader = tqdm.tqdm(test_loader)
                for (index, gaze_img, gaze_score, original_img) in tqdm_test_loader:
                    original_img = original_img.float().to(device).permute(0, 3, 1, 2)
                    gaze_img = gaze_img.to(device)
                    result = model(original_img, gaze_img, index)
                    y, cam, downsample = result
                    gpu_tracker.track()
                    for _y_item, _gaze_score_item in zip(y.cpu().detach().numpy(), gaze_score.cpu().detach().numpy()):
                        test_pred_list.append(_y_item)
                        test_gold_list.append(_gaze_score_item)
                    gpu_tracker.track()
                    loss_cam = cal_loss_cam(cam[2], gaze_img)
                    loss_score = cal_loss_score(y, gaze_score.float().cuda())
                    loss = (loss_cam + loss_score).mean()
                    tqdm_test_loader.set_description("Loss %.4f, Loss Cam %.4f, Loss Score %.4f" % (loss.data.cpu().numpy().mean(), loss_cam.data.cpu().numpy().mean(), loss_score.data.cpu().numpy().mean()))
                gpu_tracker.track()
            if use_eval and use_test:
                r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall = report_metric(valid_pred_list, valid_gold_list, test_pred_list, test_gold_list)
                tqdm_test_loader.set_description("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
                r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
                with open(os.path.join(log_dir, 'report.txt'), 'a') as f:
                    f.write("valid spearman %.4f, valid pearson %.4f, valid kendall %.4f, test spearman %.4f, test pearson %.4f, test kendall %.4f" %
                r_valid_spearman, r_valid_pearson, r_valid_kendall, r_test_spearman, r_test_pearson, r_test_kendall)
        
        # external_evaluate()    
        # np.save(os.path.join(log_dir, "checkpoint_" + str(epoch) + ".npy"), next(model.parameters()).cpu().numpy())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = inspect.currentframe()
    gpu_tracker = MemTracker(frame) # 创建显存检测对象
    logs = {
        "epoch": args.epoch
    }    
    make_log_dir(logs)

    gpu_tracker.track()
    print(args.train, args.eval, args.test)
    run(data_dir=args.data_dir, label_file=args.label_file, 
        method=args.method, epochs=args.epoch, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, 
        use_train=args.train, use_eval=args.eval, use_test=args.test, device=device)
