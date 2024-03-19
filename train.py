import os
# os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.optim import lr_scheduler
# import pickle
import cv2 as cv
from loss import SupConLoss
from models import MyDataset
from models.ccnet import ccnet
from utils import *

import copy



def test(model):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):

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

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

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

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')

# perform one epoch
def fit(epoch, model, data_loader, phase='training'):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()

    if phase == 'testing':
        # print('test')
        model.eval()

    running_loss = 0
    running_correct = 0

    for batch_id, (datas, target) in enumerate(data_loader):

        data = datas[0]
        data = data.cuda()

        data_con = datas[1]
        data_con = data_con.cuda()

        target = target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)

        loss = weight1*ce+weight2*ce2

        ## log
        running_loss += loss.data.cpu().numpy()

        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        ## update
        if phase == 'training':
            loss.backward(retain_graph=None)  #
            optimizer.step()

    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
        epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy

if __name__== "__main__" :

    parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch_num", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--com_weight",type=float,default=0.8)
    parser.add_argument("--id_num", type=int, default=378,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)

    parser.add_argument("--test_interval", type=str, default=1000)
    parser.add_argument("--save_interval", type=str, default=500)  ## 200 for Multi-spec 500 for RED

    ##Training Path
    parser.add_argument("--train_set_file", type=str, default='./data/train_Tongji.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_Tongji.txt')

    ##Store Path
    parser.add_argument("--des_path", type=str, default='./results/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/rst_test/')

    args = parser.parse_args()

    # print(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num
    weight1 = args.weight1
    weight2 = args.weight2
    comp_weight = args.com_weight

    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('weight of competition:',comp_weight)
    print('tempture:', args.temp)

    des_path = args.des_path
    path_rst = args.path_rst

    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    # path
    train_set_file = args.train_set_file
    test_set_file = args.test_set_file

    # dataset
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=128, num_workers=2, shuffle=True)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    print('------Init Model------')
    net = ccnet(num_classes=num_classes,weight=comp_weight)
    best_net = ccnet(num_classes=num_classes,weight=comp_weight)
    net.cuda()

    #
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)  ######agfzgfda

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.redstep, gamma=0.8)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    bestacc = 0

    for epoch in range(epoch_num):

        epoch_loss, epoch_accuracy = fit(epoch, net, data_loader_train, phase='training')
        # print('finish')

        val_epoch_loss, val_epoch_accuracy = fit(epoch, net, data_loader_train, phase='testing')

        scheduler.step()

        # ------------------------logs----------------------
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        # save the best model
        if epoch_accuracy >= bestacc:
            bestacc = epoch_accuracy
            torch.save(net.state_dict(), des_path + 'net_params_best.pth')
            best_net = copy.deepcopy(net)

        # save the current model and log info:
        if epoch % 10 == 0 or epoch == (epoch_num - 1) and epoch != 0:
            torch.save(net.state_dict(), des_path + 'net_params.pth')

            saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc,path_rst)

        if epoch % args.save_interval == 0:
            torch.save(net.state_dict(), des_path + 'epoch_' + str(epoch) + '_net_params.pth')

        if epoch % args.test_interval == 0 and epoch != 0:
            print('------------\n')
            test(net)

    print('------------\n')
    print('Last')
    test(net)

    print('------------\n')
    print('Best')
    test(best_net)
