import os
import cv2
import ipdb
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import pynvml

class GradCamPP():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True, device=0):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda(device=device)
        
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
    
    def __init__(self, model, conv_layer, use_cuda=True, device=0):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda(device=device)
        
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
        _cam_, _cam_np_, _pred_ = None, None, None
        for img in input:
            scores = self.model(img.unsqueeze(0))
            # class_idx = torch.argmax(scores, axis=-1)
            pred = F.softmax(scores)[0, class_idx]
            # print(class_idx, pred)
            # print(scores)
            weights = self._get_weights(class_idx, scores)
            # print(input.grad)
            # print(weights)
            cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
            
            # print(cam.shape)
            # self.clear_hooks()
            cam_np = cam.data.cpu().numpy()
            cam_np = np.maximum(cam_np, 0)
            cam_np = cv2.resize(cam_np, input.shape[2:])
            cam_np = cam_np - np.min(cam_np)
            cam_np = cam_np / np.max(cam_np)
            _cam_ = torch.cat((_cam_, cam.unsqueeze(0))) if _cam_ is not None else cam.unsqueeze(0)
            _cam_np_ = np.vstack((_cam_np_, np.expand_dims(cam_np, 0))) if _cam_np_ is not None else np.expand_dims(cam_np, 0)
            _pred_ = torch.cat((_pred_, pred.unsqueeze(0))) if _pred_ is not None else pred.unsqueeze(0)
        return _cam_, _cam_np_, _pred_

class VisualSimilarityModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1000, output_dim=5, model_choice='resnet50', 
        classifier_choice='linear', batch_size=16, layer_id='layer4', noise_type="uniform", device=None):
        super(VisualSimilarityModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_id = layer_id
        self.device = device
        self.encoder = None
        if model_choice == 'resnet34':
            self.encoder = models.resnet34(pretrained=True).to(device)
        elif model_choice == 'resnet50':
            self.encoder = models.resnet50(pretrained=True).to(device)
        elif model_choice == 'resnet101':
            self.encoder = models.resnet101(pretrained=True).to(device)
        elif model_choice == 'densenet':
            self.encoder = models.densenet(pretrained=True).to(device)
        else:
            raise 
        self.classifier = None
        if classifier_choice == "linear":
            self.classifier = nn.Linear(hidden_dim, 1).to(device)
        elif classifier_choice == "softmax":
            self.classifier = nn.Softmax(dim=0).to(device)
        else:
            raise
        self.grad_cam = GradCam(model=self.encoder, conv_layer=self.layer_id, use_cuda=True, device=self.device)
        self.embedding = nn.Linear(hidden_dim, output_dim).to(device)
        self.downsample = None
        self.nz = 16 # randomly set
        self.ngf = 32
        self.nc = 256
        # self.generate = Generator(ngpu=1, nz=self.nz, ngf=self.ngf, nc=self.nc).to(device)
        self.batch_size = batch_size
        self.noise = None
        if noise_type == "gauss":
            self.noise = torch.randn(self.batch_size, self.output_dim).to(device)
        elif noise_type == "uniform":
            self.noise = torch.rand(self.batch_size, self.output_dim).to(device)
        self.device = device

    def computecam(self, img, index, t_index=None):
        img = img / 255
        processed_img = self.preprocess_img(img)
        img = processed_img
        target_index = np.array([468,511,609,817,581,751,627], dtype=int)
        new_index = []
        for single_index in index:
            new_index.append(single_index % len(target_index))
        new_index = np.array(new_index, dtype=int)
        # ipdb.set_trace()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.total/1024**2) #总的显存大小
        # print(meminfo.used/1024**2)  #已用显存大小
        # print(meminfo.free/1024**2)  #剩余显存大小
        if t_index == None:
            # print(img.shape, self.encoder)
            ret, mask, pred = self.grad_cam(img, target_index[new_index])
            # print(meminfo.total/1024**2) #总的显存大小
            # print(meminfo.used/1024**2)  #已用显存大小
            # print(meminfo.free/1024**2)  #剩余显存大小
        else:
            # print(img.shape, self.encoder)
            ret, mask, pred = self.grad_cam(img, t_index)
            # print(meminfo.total/1024**2) #总的显存大小
            # print(meminfo.used/1024**2)  #已用显存大小
            # print(meminfo.free/1024**2)  #剩余显存大小

        return ret, pred, torch.from_numpy(mask).cuda()
    
    def preprocess_img(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        # preprocessed_img = \
        #     np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        # preprocessed_img = torch.from_numpy(preprocessed_img)
        # preprocessed_img.unsqueeze_(0)
        processed_img = preprocessed_img.requires_grad_(True)
        return processed_img

    def forward(self, x, gaze, index):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.total/1024**2) #总的显存大小
        # print(meminfo.used/1024**2)  #已用显存大小
        # print(meminfo.free/1024**2)  #剩余显存大小
        encoded = self.encoder(x)
        # Given groups=1, weight of size [64, 3, 7, 7], expected input[16, 2048, 2048, 3] to have 3 channels, but got 2048 channels instead
        # ipdb.set_trace()
        encoded_pair = self.embedding(encoded)
        klLossFunc = nn.KLDivLoss(reduction="batchmean", log_target=True)
        klLoss = klLossFunc(encoded_pair, self.noise)
        # print(meminfo.total/1024**2) #总的显存大小
        # print(meminfo.used/1024**2)  #已用显存大小
        # print(meminfo.free/1024**2)  #剩余显存大小
        
        classified = self.classifier(encoded)
        cam = self.computecam(x, index=index)
        if self.downsample is None:
            downsample = None
        else:
            downsample = self.downsample(gaze)
        noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
        # generate = self.generate(noise)
        return classified, cam, downsample, klLoss # , generate


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        """
        上卷积层可理解为是卷积层的逆运算。
        拿最后一个上卷积层举例。若卷积的输入是(nc) x 64 x 64时，
        经过Hout=（Hin+2*Padding-kernel_size）/stride+1=(64+2*1-4)/2+1=32,输出为（out_channels） x 32 x 32
        此处上卷积层为卷积层的输入输出的倒置：
        即输入通道数为out_channels，输出通道数为3；输入图片大小为（out_channels） x 32 x 32，输出图片的大小为(nc) x 64 x 64
        """
 
    def forward(self, input):
        return self.main(input)
