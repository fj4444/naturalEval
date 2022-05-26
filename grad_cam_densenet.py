import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn.functional as F
from PIL import Image
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

        # Apply pixel coefficient in each weight
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

class _GradCam():
    
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
    
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "classifier" in name.lower():
                
                out = F.relu(x, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                x = module(out)
                
            else:
                x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        # print(index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        grads_val = self.extractor.get_gradients()[-1]
        
        target = features[-1]
        target = target[0, :]

        weights = torch.mean(grads_val, axis=(2, 3))[0, :]
        cam = torch.zeros(target.shape[1:]).float().cuda()

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        # print(cam.shape)
        
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, F.softmax(output)[0, index]

class CAM_DENSENET:
    def __init__(self):
        model = models.densenet161(pretrained=True)
        # print(model)
        self.grad_cam = _GradCam(model=model, conv_layer='features', use_cuda=True)
        # self.grad_cam = GradCam(model=model, feature_module=model.features, \
        #                target_layer_names=["denseblock4"], use_cuda=True)
        self.log_dir = "./"
        
    def __call__(self, img, index, log_dir):
        self.log_dir = log_dir
        img = img / 255
        raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        input = self.preprocess_image(img)
        target_index = [468,511,609,817,581,751,627]
        ret, mask, pred = self.grad_cam(input, target_index[index % len(target_index)])
        # print(img.shape)
        self.show_cam_on_image(raw_img, mask)
        return ret, pred
        
    def preprocess_image(self, img):
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
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam_densenet.jpg'))
        # Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))


