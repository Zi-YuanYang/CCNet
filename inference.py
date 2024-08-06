
import os
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models


# print(os.getcwd())
import pickle
import numpy as np
from PIL import Image
import cv2 as cv
from loss import SupConLoss

import matplotlib.pyplot as plt

from utils.util import plotLossACC, saveLossACC, saveGaborFilters, saveParameters, saveFeatureMaps

plt.switch_backend('agg')

from models import MyDataset
from models.ccnet import ccnet
from utils import *

import copy
import argparse


### This is for Original CompNet

parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

parser.add_argument("--batch_size",type=int,default = 2048)
parser.add_argument("--epoch_num",type=int,default = 3000)
parser.add_argument("--temp", type=float, default= 0.07)
parser.add_argument("--weight1",type=float,default = 0.8)
parser.add_argument("--weight2",type=float,default = 0.2)
parser.add_argument("--id_num",type=int, default = 600, help = "IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 No_Delete_PolyU 386 Tongji_LR 300")
parser.add_argument("--source_id_num",type=int, default = 600, help = "IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 No_Delete_PolyU 386 Tongji_LR 300")
parser.add_argument("--gpu_id",type=str, default='0')
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument("--redstep",type=int, default=300)
parser.add_argument("--weight_chan",type=float,default=0.8,help="The weight of channel competition branch")

parser.add_argument("--mode",type = str,default='large_mul',help='Tiny, Middle, Large, Four, normal')

parser.add_argument("--test_interval",type=str,default = 1000)
parser.add_argument("--save_interval",type=str,default = 1000)  ## 200 for Multi-spec 500 for RED

##Training Path
parser.add_argument("--train_set_file",type=str,default='./data/train_all_server.txt')
parser.add_argument("--test_set_file",type=str,default='./data/test_server.txt')

##Store Path
parser.add_argument("--des_path",type=str,default='/data/YZY/Palm_DOC/Tongji_add/checkpoint/')
parser.add_argument("--path_rst",type=str,default='/data/YZY/Palm_DOC/Tongji_add/rst_test/')
parser.add_argument("--check_point",type=str, default='/data/YZY/Palm_Doc/')
# parser.add_argument("--save_path",type=str,default='./cross-db-checkpoint/PolyU_1')

args = parser.parse_args()

# print(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

batch_size = args.batch_size
num_classes = args.id_num  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200 POLYU 378 Multi-Spa 500

# des_path = './original_result/checkpoint_test/'

# path
train_set_file = args.train_set_file
test_set_file = args.test_set_file

path_rst = args.path_rst

if not os.path.exists(path_rst):
    os.makedirs(path_rst)

trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)

data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0, shuffle=True)
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0, shuffle=False)

net = ccnet(num_classes=num_classes,weight=args.weight_chan)
net.load_state_dict(torch.load(args.check_point),strict=False)


def test(model):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    ### Calculate EER

    path_hard = os.path.join(path_rst, 'rank1_hard')

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 1024  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    # num_classes = 600  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200
    net = model

    # device = torch.device("cuda")
    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        # break

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    assert num_training_samples % classNumel == 0
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction for test set.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nfeature extraction done!')
    print('\n\n')

    print('start feature matching ...\n')

    print('Verification EER of the test set ...')

    print('Start EER for Train-Test Set ! Its wrong? \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

test(net)

