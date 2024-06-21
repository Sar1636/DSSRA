import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torch.optim.optimizer import Optimizer
import scipy.io as sio
from model import ANN,DSSRA
from loss import NormalizedCrossEntropy, ReverseCrossEntropy, NFLandRCE, NCEandRCE,NormalizedFocalLoss,NCEandMAE
import Data_generator
from Draw_feature_map import draw_feature_map
from utils import evalution
from torchstat import stat
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import mat73
import tifffile
import time

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int,
                    default=16, help='input batch size')
parser.add_argument('--max_iter', type=int, default=100,
                    help='max training iterations')
parser.add_argument('--iters', type=int, default=10,
                    help='Experiments iterations')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pathDict = {('./datasets/KSC/KSC.mat',
#              './datasets/KSC/KSC_gt.mat'): (20, range(4, 13, 4))}

# pathDict = {('./datasets/salinas/Salinas.mat',
#              './datasets/salinas/Salinas_gt.mat'): (24, range(4, 16, 4))}

pathDict = {('./datasets/paviaU/PaviaU.mat',
             './datasets/paviaU/PaviaU_gt.mat'): (52, range(1,9,1))}

# pathDict = {('./datasets/Pavia_Centre/Pavia.mat',
#              './datasets/Pavia_Centre/Pavia_gt.mat'): (52, range(4,9,4))}

# pathDict = {('./datasets/Botswana/Botswana.mat',
#              './datasets/Botswana/Botswana_gt.mat'): (52, range(4,14,4))}

# pathDict = {('./datasets/Dioni/Dioni.mat',
#              './datasets/Dioni/Dioni_gt_out68.mat'): (42, range(4,12,4))}


total_training_time = 0
for (dataset, gt) in pathDict.keys():

    (CL_num, NL_list) = pathDict[(dataset, gt)]
    Acc_total = []
    AA_total = []
    kappa_total = []
    model_best = {}

    for i in range(opt.iters):
        start_time = time.time()

        Acc_list = []
        AA_list = []
        kappa_list = []
        Loss = []
        data = Data_generator.DataGenerator(dataset,
                                            gt,
                                            hwz=3,
                                            CL_num=CL_num,
                                            NL_num=8)  # add
        data.start()
        for NL_num in NL_list:
            data.add_noise(NL_num - data.NL_num)
            data.getshape()
            model = DSSRA(input_channels=data.in_channel,
                         patch_size=data.hwz * 2 + 1,
                         n_classes=data.class_num).to(device)

            nce = NormalizedCrossEntropy(data.class_num).to(device)
            rce = ReverseCrossEntropy(data.class_num).to(device)
            nflrce = NFLandRCE(data.class_num).to(device)  # add
            loss1 = NormalizedFocalLoss(data.class_num).to(device)
            ncemae = NCEandMAE(data.class_num).to(device)  # add
            ncerce = NCEandRCE(data.class_num).to(device)  # add
            ce = nn.CrossEntropyLoss().to(device)
            Optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

            train, test = data.to_tensor()
            trainloader = DataLoader(
                train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
            testloader = DataLoader(
                test, batch_size=16, shuffle=True, drop_last=False)

            for epoch in range(opt.max_iter):
                training_loss = 0
                for i,(info, fake_label, true_label) in enumerate (trainloader):
                    Optimizer.zero_grad()
                    info = info.unsqueeze(1)
                    info = info.to(device, torch.float32)
                    fake_label = fake_label.to(device, torch.long)
                    res = model(info)
                    score = torch.ones_like(fake_label)
                    loss = nflrce(res, fake_label)
                    training_loss += loss.item()
                    loss.backward()
                    Optimizer.step()

                Loss.append(training_loss / len(trainloader))
                if epoch > 1 and abs(Loss[-1] - Loss[-2]) < 1e-4:
                    print('break at epoch {}'.format(epoch))
                    break

            Acc, aa, kappa = evalution(testloader, model, data.class_num)
            Acc_list.append(Acc)
            AA_list.append(aa)
            kappa_list.append(kappa)
            if NL_num not in model_best:
                model_best[NL_num] = (model, data, Acc)
                print('Best model initialize to {:.2f} when NL_num is {}'.format(
                    Acc * 100, NL_num))
            else:
                if Acc > model_best[NL_num][2]:
                    Acc_before = model_best[NL_num][2]
                    model_best[NL_num] = (model, data, Acc)
                    print('Best model update from {:.2f} to {:.2f} when NL_num is {}'.format(
                        Acc_before * 100, Acc * 100, NL_num))

        Acc_total.append(Acc_list)
        AA_total.append(AA_list)
        kappa_total.append(kappa_list)
        print(Acc_total)
    print('OA of classification is {},\n AA of classification is {},\n kappa of classification is {}'.format(np.mean(Acc_total, 0) * 100,
                                                                                                             np.mean(
                                                                                                              AA_total, 0) * 100,
                                                                                                             np.mean(kappa_total, 0) * 100))
    AA_mean = np.mean(np.mean(AA_total, 0), 1)
    print('AA is {}'.format(AA_mean))
    AA_std = []
    for i in range(len(AA_mean)):
        AA_temp = [AA_total[j][i] for j in range(len(AA_total))]
        AA_std.append(np.std(np.mean(AA_temp, 1)) * 100)
    print('OA std is {},\n AA std is {},\n kappa std is {}'.format(np.std(Acc_total, 0) * 100,
                                                                   AA_std,
                                                                   np.std(kappa_total, 0) * 100))

    

    matfn1 = sio.loadmat("./datasets/paviaU/PaviaU.mat")
    matfn2 = sio.loadmat('./datasets/paviaU/PaviaU_gt.mat')

    # matfn1 = sio.loadmat("./datasets/Pavia_Centre/Pavia.mat")
    # matfn2 = sio.loadmat('./datasets/Pavia_Centre/Pavia_gt.mat')

    # matfn1 = sio.loadmat("./datasets/Botswana/Botswana.mat")
    # matfn2 = sio.loadmat('./datasets/Botswana/Botswana_gt.mat')

    # matfn1 = sio.loadmat("./datasets/Dioni/Dioni.mat")
    # matfn2 = sio.loadmat('./datasets/Dioni/Dioni_gt_out68.mat')



    for key in model_best.keys():
 
        draw_feature_map('pavia_U', model_best[key][0], model_best[key][1])