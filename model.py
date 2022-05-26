import torch
import torch.nn as nn
import cv2
from torchvision import models
import torch
import numpy as np
import warnings
import ipdb
warnings.filterwarnings("ignore")
from torch.autograd import Function
import torch.nn.functional as F
from PIL import Image
import ipdb
import os

class GradCamPP():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        grad_2 = self.hook_g.pow(2)
        grad_3 = self.hook_g.pow(3)
        alpha = grad_2 / (1e-13 + 2 * grad_2 + (grad_3 * self.hook_a).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        pred = F.softmax(scores)[0, class_idx]
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class GradCam():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        # class_idx = torch.argmax(scores, axis=-1)
        pred = F.softmax(scores)[0, class_idx]
        # print(class_idx, pred)
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred


class CAM:
    
    def __init__(self, model, conv_layer, log_dir):
        # model = models.resnet50(pretrained=True)
        # self.grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        # self.log_dir = "./"
        self.grad_cam = GradCam(model=model, conv_layer=conv_layer, use_cuda=True)
        self.log_dir = log_dir
        
    def __call__(self, img, t_index=None):
        self.t_index = t_index
        img = img / 255
        # raw_img = img[:, :255, :255, :].data.cpu().numpy()[0].transpose((1, 2, 0))
        input = self.preprocess_image(img.detach().cpu())
        input = torch.from_numpy(input[:, :, :, :].detach().numpy().transpose((0, 3, 1, 2))).cuda()
        ret, _, pred = self.grad_cam(input, t_index)
        # print(img.shape)
        # self.show_cam_on_image(raw_img, mask)
        return input, ret, pred
        
    def preprocess_image(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, :, :, i] = preprocessed_img[:, :, :, i] - means[i]
            preprocessed_img[:, :, :, i] = preprocessed_img[:, :, :, i] / stds[i]
        # preprocessed_img = 
        #     np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        # preprocessed_img = torch.from_numpy(preprocessed_img)
        # preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        ipdb.set_trace()
        # if self.t_index==None:
        #     Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
        #     Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
        #     
        # else:
        #     Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam_'+str(self.t_index)+'.jpg'))
        # Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))


class GazeModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size=4, feature_dim=16, class_num=7):
        super(GazeModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(500, 500))
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=(300, 300))
        # self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(100, 100))
        # self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(50, 50))
        self.linear_1 = nn.Linear(in_features=batch_size, out_features=3)
        self.encoder  =  nn.Sequential(
            nn.Linear(224 * 224, 112 * 112),
            nn.Tanh(),
            nn.Linear(112 * 112, 64 * 64),
            nn.Tanh(),
            nn.Linear(64 * 64, 32 * 32),
            nn.Tanh(),
            nn.Linear(32 * 32, 16 * 16),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 3, feature_dim),
        ).cuda()
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        self.linear = nn.Linear(in_features = feature_dim, out_features = feature_dim, bias = True).cuda()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.prior = torch.normal(0, 1, size=(batch_size, feature_dim)).cuda()
        self.resnet_model = models.resnet50(pretrained=True).cuda()
        self.cam1 = CAM(self.resnet_model, 'layer4', './demo1')
        self.classification = nn.Linear(in_features=feature_dim, out_features=class_num, bias=True).cuda()

    def forwarding(self, input_img):
        # ipdb.set_trace()
        print("begin forwarding ...\n")
        input_transposed = torch.from_numpy(input_img.detach().cpu().numpy().transpose((0, 3, 1, 2)))
        input_transposed = input_transposed.view(self.batch_size, 3, 224 * 224)
        print(input_transposed.shape)
        # ipdb.set_trace()
        input1 = []
        ret1 = []
        pred1 = []
        for idx in range(input_img.shape[0]):
            _input, _ret, _pred = self.cam1(input_img[idx].unsqueeze(dim=0))
            input1.append(_input)
            ret1.append(_ret)
            pred1.append(_pred)
        # ipdb.set_trace()
        input1 = torch.stack(input1, dim=0).squeeze(1)
        ret1 = torch.stack(ret1, dim=0).squeeze(1)
        pred1 = torch.stack(pred1, dim=0).squeeze(1)
        encoded = self.encoder(input_transposed.cuda())
        result = torch.softmax(self.classification(encoded), dim=1)
        compare_img = cv2.resize(cv2.imread("./data/eye-gazing.png"), (224, 224))
        compare_img = compare_img / 255
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        for i in range(3):
            compare_img[:, :, i] = compare_img[:, :, i] - means[i]
            compare_img[:, :, i] = compare_img[:, :, i] / stds[i]

        picture1 = input1.detach().cpu().numpy().transpose((0, 2, 3, 1))
        picture2 = np.stack((compare_img))
        loss1 = ((picture1 - picture2) ** 2).mean()
        pairing = self.linear(encoded)
        loss = self.kl_loss(self.prior, pairing)        
        return encoded, result, loss, loss1
    
    def __call__(self, input_img):
        return self.forwarding(input_img)
