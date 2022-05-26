import cv2
from grad_cam import CAM
from torchvision import models
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = models.resnet50(pretrained=True).cuda()
Cam1 = CAM(model, 'layer4', './demo1')
Cam2 = CAM(model, 'layer4', './demo2')
model = models.densenet161(pretrained=True).cuda()
Cam3 = CAM(model, 'features', './demo3')

# 图片shape=(1, 3, 224, 224)
img = torch.from_numpy(np.transpose(np.resize(cv2.imread('./test.jpg'), (224, 224, 3)), (2, 0, 1))).cuda().unsqueeze(0)

# 如果要指定针对哪个index
target_index = 468  # cab
print(Cam1(img, target_index))

# 或者不指定，则生成score最高的
print(Cam2(img))

# 理论上只要给出model和针对的layer，都能做grad_cam
# 但是实测时感觉classification的model最好用
# 返回的是原始saliency map和指定index的pred（或不指定时的最大pred）
ret, pred = Cam3(img, target_index)
print(ret, pred)
