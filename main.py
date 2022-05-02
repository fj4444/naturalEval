from cv2 import Param_ALGORITHM
from model import GazeModel
import torch
import numpy as np
import cv2
import time
import torch.utils.data as Data
import torch.nn.functional as F
from PIL import Image
import argparse
import ipdb
import os
import os.path
import numpy as np
from regex import D
import joblib
## for cluster_gcn.py
import torch


class Logger(object):
    def __init__(self, runs, info=None, log=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.log = log

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print(self, mess):
        if self.log is not None:
            self.log.info(mess)
        else:
            print(mess)

    def print_statistics(self, run=None):
        if run is not None:
            if self.results[run] == []:
                self.print(f'WARNING: results from run {run} not recorded')
                return
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            self.print(f'Run {run + 1:02d}:')
            self.print(f'Highest Train: {result[:, 0].max():.2f}')
            self.print(f'Highest Valid: {result[:, 1].max():.2f}')
            self.print(f'  Final Train: {result[argmax, 0]:.2f}')
            self.print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            results_recorded = [line for line in self.results if line != []]
            if len(results_recorded) < len(self.results):
                self.print(f'WARNING: results from only {len(results_recorded)}/{len(self.results)} runs are recorded')
            result = 100 * torch.tensor(results_recorded)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            self.print(f'All runs:')
            r = best_result[:, 0]
            self.print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            self.print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            self.print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            self.print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


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

class DATASET(torch.utils.data.Dataset):
    def __init__(self, root, batch_size=1, train=False, input_size=224, **kwargs):
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
                self.picture_list = [cv2.resize(cv2.imread(os.path.join(root, picture_name.strip().split(" ")[0])), (224, 224)) for picture_name in picture_name_list]
                # ipdb.set_trace()
                with open(self.train_label_name_file, "r") as picture_labels:
                    self.label_list = [int(str(x).strip("\n")) for x in picture_labels.readlines()]
                
        else:
            with open(self.val_picture_name_file, "r") as picture_names:
                picture_name_list = picture_names.readlines()
                self.picture_list = [cv2.resize(cv2.imread(os.path.join(root, picture_name.strip().split(" ")[0])), (224, 224)) for picture_name in picture_name_list]
                # ipdb.set_trace()
                with open(self.train_label_name_file, "r") as picture_labels:
                    self.label_list = [int(str(x).strip("\n")) for x in picture_labels.readlines()]

        # self.data_dict = joblib.load(pkl_file)
        self.batch_size = batch_size
        self.idx = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample * 1.0 / self.batch_size))
    
    @property 
    def n_sample(self):
        print(len(self.picture_list))
        return len(self.picture_list)
    
    def __len__(self):
        return self.n_batch 

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.idx >= self.n_batch:
            self.idx = 0
            raise StopIteration 
        else:
            img = self.picture_list[self.idx * self.batch_size: (self.idx + 1) * self.batch_size].astype('float32')
            target = self.label_list[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
            self.idx += 1
            return img, target
    
    def __getitem__(self, index):
        # ipdb.set_trace()
        img = torch.from_numpy(np.array(self.picture_list[index * self.batch_size: (index + 1) * self.batch_size]))
        target = torch.from_numpy(np.array(self.label_list[index * self.batch_size: (index + 1) * self.batch_size]))
            
        return img.float(), target.long() # Don't know whether this is right



def train(model, loader, optimizer, device):
    model.train()
    total_loss = total_examples = 0
    total_correct = total_examples = 0
    for read_cnt, (data, label) in enumerate(loader):
        for idx in range(0, data.shape[0]):
            # if data.train_mask.sum() == 0:
            #   continue
            optimizer.zero_grad()
            ipdb.set_trace()
            _, result, loss, loss1 = model(data[idx][:1].cuda())
            loss.backward(retain_graph=True)
            loss1.backward(retain_graph=True)
            y = label[idx].squeeze(1)
            ipdb.set_trace()
            loss2 = F.nll_loss(result, y)
            loss2.backward(retain_graph=True)
            optimizer.step()

            num_examples = data[idx].shape[0] # data.train_mask.sum().item()
            total_loss += loss.item() * num_examples 
            total_examples += num_examples 

            total_correct += out.argmax(dim=-1).eq(y).sum().item()
            total_examples += y.size(0)
    
    return total_loss / total_examples, total_correct / total_examples

def eval(model, loader_valid, loader_test, device):
    model.eval()
    y_true_valid, y_true_test = [], []
    y_pred_valid, y_pred_test = [], []
    for (data_valid, data_test) in zip(loader_valid, loader_test):
        data_valid = data_valid.to(device)
        data_test = data_test.to(device)
        # if data.test_mask.sum() == 0:
        #   continue
        out_valid = model(data_valid)
        y_valid = data_valid.y.squeeze(1)
        out_test = model(data_test)
        y_test = data_test.y.squeeze(1)
        y_true_valid.append(y_valid)
        y_pred_valid.append(out_valid.argmax(dim=-1, keepdim=True))
        y_true_test.append(y_test)
        y_pred_test.append(out_test.argmax(dim=-1, keepdim=True))
    y_true_valid = torch.cat(y_true_valid).unsqueeze(-1)
    y_pred_valid = torch.cat(y_pred_valid)
    y_true_test = torch.cat(y_true_test).unsqueeze(-1)
    y_pred_test = torch.cat(y_pred_test)
    train_acc = -1
    # valid_acc and test_acc to be implemented
    return train_acc, 0, 0

def main():
    parser = argparse.ArgumentParser(description='Naturalness Evaluation')
    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=3) # change from 10 to 3
    parser.add_argument('--dropouter', type=float, default=0.5)
    parser.add_argument('--eval_loader', type=str, choices=['cluster', 'ns'], default="cluster")
    
    args = parser.parse_args()
    print(args)

    # device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = GazeModel(num_embeddings=100, embedding_dim=300, batch_size=1, feature_dim=16).cuda()
    train_dataset, eval_dataset = get(batch_size=10, data_root='.', train=True, val=True)
    
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size, 
    )
    eval_loader = Data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
    )
    ipdb.set_trace()
    logger = Logger(args.runs, args)
    # evaluator = Evaluator()
    # for idx in range(0, 100):
    #     img = torch.from_numpy(np.transpose(np.resize(cv2.imread('./demo_img/img' + str(idx) + '.png'), (224, 224, 3)), (2, 0, 1))).cuda().unsqueeze(0)
    #### main
    for run in range(args.runs):
        time_start_run = time.time()
        # 重置模型的一些参数
        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(1, 1 + args.epochs):
            ## train
            loss, train_acc = train(model, train_loader, optimizer, device)
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, ',
                      f'Approx Train Acc: {train_acc:.4f}')
            ## eval
            if epoch > -1 and epoch % args.eval_steps == 0:
                result = eval(model, eval_loader, device)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}'
                      f'Epoch: {epoch:02d}'
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        time_end_run = time.time()
        print("---------------------------------------------------\n")
        print("Total time : ", (time_end_run - time_start_run) / 3600 , "(h) ", 
                ((time_end_run - time_start_run) % 3600) / 60, "(m) ", 
                ((time_end_run - time_start_run) % 60), "(s) \n")
        print("---------------------------------------------------\n")
        logger.print_statistics(run)
    
    logger.print_statistics(run)

if __name__ == "__main__":
    main()
