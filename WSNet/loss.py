import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()

    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())

        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)

        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)

        return loss_img + loss_edge


class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss