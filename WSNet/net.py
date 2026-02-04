from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model.WSNet.WSNet import *
# from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, dataset_name ,mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()

        if model_name == 'WSNet' and (dataset_name != 'NUDT-SIRST'):
            self.model = WSNet()
        else:
            self.model = WSNet_Large()

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss



