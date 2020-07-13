import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 0.01):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(-1).sum(-1).sum(-1)
    union = (pred + target).sum(-1).sum(-1).sum(-1)

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = (1 - dice)
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.25):
	bce = F.binary_cross_entropy_with_logits(pred, target)

	pred = F.sigmoid(pred)
	dice = dice_loss(pred, target)

	loss = bce * bce_weight + dice * (1 - bce_weight)

	metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
	metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

	return loss, dice