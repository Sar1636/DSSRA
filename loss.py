import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


'''
Cite from Ma et al.
'''
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, gamma=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.gamma = gamma

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        RCE = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        # rce= torch.pow((1 - torch.exp(-self.gamma * RCE)), 1/self.gamma)
        rce = self.scale *RCE.mean()
        return rce


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, gamma=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.gamma = gamma

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)  # add
        NCE = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        nce = self.scale * NCE.mean()
        return nce


class NCEandRCE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, scale=1.0, beta=0.1, gamma=0.48):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scale = scale
        self.nce = NormalizedCrossEntropy(
            scale=beta, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(
            scale=beta, gamma=gamma, num_classes=num_classes)

    def forward(self, pred, labels):
        ncerce = (self.gamma)*self.nce(pred, labels) +\
            (1-self.gamma)*self.rce(pred, labels)
        return ncerce

########################################################################################
class SCELoss(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.1):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        sce = self.alpha * ce + self.beta * rce.mean()
        return sce


class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        dmi = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
        return dmi


class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, num_classes=16, scale=1.0, gamma=0, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale
        self.alpha = alpha

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp())
                               ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss1 = -1 * (1-pt)**self.gamma * logpt
        loss1 = self.scale * loss1 / normalizor

        if self.size_average:
            return loss1.mean()
        else:
            return loss1.sum()


class NFLandRCE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.54): #56
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=1.0, gamma=1.0, num_classes=num_classes)
        # self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        nflrce = (1-self.gamma)*self.nfl(pred, labels) + self.gamma*self.rce(pred, labels)
        # nflrce = self.gamma*self.nfl(pred, labels)
        return nflrce


class NMAE(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NMAE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        normalizor = 1 / (2 * (self.num_classes - 1))
        # mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        mae = torch.abs(pred - label_one_hot).sum(dim=1)
        return self.scale * normalizor * mae.mean()
    

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        # mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        mae = torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()
    
class NCEandMAE(torch.nn.Module):
    def __init__(self,num_classes, alpha=1.0, beta=1.0, gamma=1):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.nce = nn.CrossEntropyLoss()
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        ncemae = self.nce(pred, labels) + self.mae(pred, labels)
        return ncemae

