import numpy as np
import torch
import os
import pickle
from PIL import Image
import cv2

import matplotlib.pyplot as plt
plt.switch_backend('agg')




def getFileNames(txt):
    '''
    description: parse the xxx.txt (image_path+' '+label) file::
    input: path of the .txt file\n
    return: a list of DB image pathes
    '''
    fileDB = []
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(' ')
            fileDB.append(item[0])
    return fileDB






def saveimgs(act, dir='featImgs', epoch=0):
    '''
    description: save batched feature maps (GPU->CPU) into one figure using PIL Image\n
    act: activation values obtained from the network layer\n
    dir: output folder (will be automatically created)\n
    epoch: the epoch ID\n
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

    # when input more than one layer
    # for id, act in enumerate(acts):

    act = act.detach().cpu().numpy()

    b, c, h, w = act.shape
    stp = max(w//20, 1)

    imgs = np.ones((b*h+(b+1)*stp, c*w+(c+1)*stp), dtype=np.uint8)*255

    # batch and channels
    for bid in range(b):
        for cid in range(c):
            img = act[bid, cid, :, :]
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255

            srow = stp*(bid+1)
            scol = stp*(cid+1)
            imgs[h*bid + srow : h*(bid+1) + srow, w*cid + scol : w*(cid+1)+ scol] = img.astype(np.uint8)


    im = Image.fromarray(imgs.astype("uint8")).convert('L')

    im.save(os.path.join(dir, 'layer'+('_ep%04d'%epoch)+'.png'))
    # im.save(os.path.join(dir, 'layer'+str(id)+('_ep%04d'%epoch)+'.png'))
    # cv2.imwrite(os.path.join(dir, 'layer'+str(id)+('_ep%04d'%epoch)+'.png'), imgs)
    # img = np.array(img, dtype=np.uint8)




def saveimgs2(act, dir='featImgs', epoch=0):
    '''
    description: save batched feature maps (GPU->CPU) into one figure using plt.figure() subplots\n
    act: activation values obtained from the network layer\n
    dir: output folder (will be automatically created)\n
    epoch: the epoch ID\n
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

    act = act.detach().cpu().numpy()

    b, c, h, w = act.shape

    fig = plt.figure(figsize=(100, int(100.*b/c)))#
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

    for bid in range(b): # batch id
        for cid in range(c): # channel id
            ax = fig.add_subplot(b, c, bid*c+cid+1, xticks=[], yticks=[])
            ax.imshow(act[bid, cid, :, :])
    plt.savefig(os.path.join(dir, 'layer'+('_ep%04d'%epoch)+'.png'))





def saveimgs3(netOut, name='feat', featflag=False):
    '''
    save feature maps via global normalization
    '''
    imgs = netOut.data

    imgs = torch.cat(torch.split(imgs, 1, dim=1), dim=3)
    imgs = torch.cat(torch.split(imgs, 1, dim=0), dim=2)
    imgs = torch.squeeze(imgs)
    imgs = imgs.numpy()

    imgs = (imgs-imgs.min())/(imgs.max()-imgs.min()+1e-8)*255
    cv2.imwrite(name+'.bmp', imgs.astype(np.uint8))
    # imgs = Image.fromarray(imgs.numpy())
    # imgs.save(name+'.jpg')


class RegLayers():
    features = []
    def __init__(self, net):
        self.hooks = []

        self.hooks.append(net.cb1.gabor_conv2d.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb1.argmax.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb1.conv1.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb1.conv2.register_forward_hook(self.hook_fn))

        self.hooks.append(net.cb2.gabor_conv2d.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb2.argmax.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb2.conv1.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb2.conv2.register_forward_hook(self.hook_fn))

        self.hooks.append(net.cb3.gabor_conv2d.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb3.argmax.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb3.conv1.register_forward_hook(self.hook_fn))
        self.hooks.append(net.cb3.conv2.register_forward_hook(self.hook_fn))

    def hook_fn(self, model, input, output):
        self.features.append(output)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def extract_layers(model, input):

    la = RegLayers(model)

    la.features = []

    model.eval()

    o = model(input)

    la.remove()

    acts = la.features
    return acts






def saveFeatureMaps(net, data_loader, epoch):
    data, _ = iter(data_loader).next()
    device = net.fc.weight.device

    saveimgs(data, dir='./original_result/rst/images/00_input_Img',epoch=epoch)

    acts = extract_layers(net, data.to(device))
    saveimgs(acts[0], dir='./original_result/rst/images/01_cb1_gabor',epoch=epoch)
    saveimgs(acts[1], dir='./original_result/rst/images/02_cb1_scc',epoch=epoch)
    saveimgs(acts[2], dir='./original_result/rst/images/03_cb1_conv1',epoch=epoch)
    saveimgs(acts[3], dir='./original_result/rst/images/04_cb1_conv2',epoch=epoch)
    saveimgs(acts[4], dir='./original_result/rst/images/05_cb2_gabor',epoch=epoch)
    saveimgs(acts[5], dir='./original_result/rst/images/06_cb2_scc',epoch=epoch)
    saveimgs(acts[6], dir='./original_result/rst/images/07_cb2_conv1',epoch=epoch)
    saveimgs(acts[7], dir='./original_result/rst/images/08_cb2_conv2',epoch=epoch)
    saveimgs(acts[8], dir='./original_result/rst/images/09_cb3_gabor',epoch=epoch)
    saveimgs(acts[9], dir='./original_result/rst/images/10_cb3_scc',epoch=epoch)
    saveimgs(acts[10], dir='./original_result/rst/images/11_cb3_conv1',epoch=epoch)
    saveimgs(acts[11], dir='./original_result/rst/images/12_cb3_conv2',epoch=epoch)


def saveGaborFilters(net, epoch):
    '''
    save the learned Gabor filters of LGC
    '''
    if not os.path.exists('./original_result/rst/images/gaborfilters'):
        os.makedirs('./original_result/rst/images/gaborfilters')

    kernel1 = net.cb1.gabor_conv2d.kernel
    channel_in = net.cb1.channel_in
    channel_out = net.cb1.n_competitor

    kernel = kernel1.detach().cpu().numpy()
    for o in range(channel_out):
        for i in range(channel_in):
            ws = kernel[o, i, :, :]
            # plt.matshow(ws, cmap='gist_gray')
            # plt.savefig('gaborfilters/%03d.png'%o )
            # plt.show()
            img = ws
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255
            im = Image.fromarray(img.astype("uint8")).convert('L')
            im.save('./original_result/images/gaborfilters/%d_cb1_%03d.png'%(epoch, o))

    kernel2 = net.cb2.gabor_conv2d.kernel
    channel_in = net.cb2.channel_in
    channel_out = net.cb2.n_competitor

    kernel = kernel2.detach().cpu().numpy()
    for o in range(channel_out):
        for i in range(channel_in):
            ws = kernel[o, i, :, :]
            img = ws
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255
            im = Image.fromarray(img.astype("uint8")).convert('L')
            im.save('./original_result/images/gaborfilters/%d_cb2_%03d.png'%(epoch, o))

    kernel3 = net.cb3.gabor_conv2d.kernel
    channel_in = net.cb3.channel_in
    channel_out = net.cb3.n_competitor

    kernel = kernel3.detach().cpu().numpy()
    for o in range(channel_out):
        for i in range(channel_in):
            ws = kernel[o, i, :, :]
            img = ws
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255
            im = Image.fromarray(img.astype("uint8")).convert('L')
            im.save('./original_result/images/gaborfilters/%d_cb3_%03d.png'%(epoch, o))




def printParameters(net):
    '''
    print the learable parameters of the Competitive Block
    '''
    if net.fc.weight.device == 'cpu':
        print('------learnable parameters------')
        print('cb1-(a,b): ', net.cb1.a, '\t', net.cb1.b)
        print('cb2-(a,b): ', net.cb2.a, '\t', net.cb2.b)
        print('cb3-(a,b): ', net.cb3.a, '\t', net.cb3.b)

        print('cbs, sigma, gamma, f:')
        print('cb1 %.6f \t%.6f \t%.6f'%(net.cb1.gabor_conv2d.sigma, net.cb1.gabor_conv2d.gamma, net.cb1.gabor_conv2d.f))
        print('cb2 %.6f \t%.6f \t%.6f'%(net.cb2.gabor_conv2d.sigma, net.cb2.gabor_conv2d.gamma, net.cb2.gabor_conv2d.f))
        print('cb3 %.6f \t%.6f \t%.6f'%(net.cb3.gabor_conv2d.sigma, net.cb3.gabor_conv2d.gamma, net.cb3.gabor_conv2d.f))
        print('--------------------------------')
    else:
        print('------learnable parameters------')
        print('cb1-(a,b): ', net.cb1.a.item(), '\t', net.cb1.b.item())
        print('cb2-(a,b): ', net.cb2.a.item(), '\t', net.cb2.b.item())
        print('cb3-(a,b): ', net.cb3.a.item(), '\t', net.cb3.b.item())

        print('cbs, sigma, gamma, f:')
        print('cb1 %.6f \t%.6f \t%.6f'%(net.cb1.gabor_conv2d.sigma.item(),  net.cb1.gabor_conv2d.gamma.item(), net.cb1.gabor_conv2d.f.item()))
        print('cb2 %.6f \t%.6f \t%.6f'%(net.cb2.gabor_conv2d.sigma.item(), net.cb2.gabor_conv2d.gamma.item(), net.cb2.gabor_conv2d.f.item()))
        print('cb3 %.6f \t%.6f \t%.6f'%(net.cb3.gabor_conv2d.sigma.item(), net.cb3.gabor_conv2d.gamma.item(), net.cb3.gabor_conv2d.f.item()))
        print('--------------------------------')


def saveParameters(net, epoch):
    path_rst = './original_result/'
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    with open(os.path.join(path_rst,'param_gabor_ab.txt'), 'a+') as f:
        f.write('\n------learnable parameters @ epoch-%04d------\n'%epoch)
        f.write('cb1-(a,b): %.4f,\t%.4f\n'%(net.cb1.a.detach().cpu().numpy(), net.cb1.b.detach().cpu().numpy()))
        f.write('cb2-(a,b): %.4f,\t%.4f\n'%(net.cb2.a.detach().cpu().numpy(), net.cb2.b.detach().cpu().numpy()))
        f.write('cb3-(a,b): %.4f,\t%.4f\n'%(net.cb3.a.detach().cpu().numpy(), net.cb3.b.detach().cpu().numpy()))

        f.write('cbs, sigma, gamma, f:\n')
        f.write('cb1 %.6f \t%.6f \t%.6f\n'%(net.cb1.gabor_conv2d.sigma.detach().cpu().numpy(), net.cb1.gabor_conv2d.gamma.detach().cpu().numpy(), net.cb1.gabor_conv2d.f.detach().cpu().numpy()))
        f.write('cb2 %.6f \t%.6f \t%.6f\n'%(net.cb2.gabor_conv2d.sigma.detach().cpu().numpy(), net.cb2.gabor_conv2d.gamma.detach().cpu().numpy(), net.cb2.gabor_conv2d.f.detach().cpu().numpy()))
        f.write('cb3 %.6f \t%.6f \t%.6f\n'%(net.cb3.gabor_conv2d.sigma.detach().cpu().numpy(), net.cb3.gabor_conv2d.gamma.detach().cpu().numpy(), net.cb3.gabor_conv2d.f.detach().cpu().numpy()))
        f.write('--------------------------------\n\n')


def plotLossACC(train_losses, val_losses, train_accuracy, val_accuracy):
    path_rst = './original_result/'
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b', label='training loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='test loss')
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.savefig(os.path.join(path_rst, 'losses.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'b', label='training accuracy')
    plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r', label='test accuracy')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch number')
    plt.ylabel('accuracy (%)')
    plt.savefig(os.path.join(path_rst, 'accuracy.png'))
    plt.close()


def saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc, path_rst):
    # path_rst = './original_result/'
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    # save as pickle
    with open(os.path.join(path_rst,'train_losses.pickle'), 'wb') as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(path_rst,'val_losses.pickle'), 'wb') as f:
        pickle.dump(val_losses, f)
    with open(os.path.join(path_rst,'train_accuracy.pickle'), 'wb') as f:
        pickle.dump(train_accuracy, f)
    with open(os.path.join(path_rst,'val_accuracy.pickle'), 'wb') as f:
        pickle.dump(val_accuracy, f)


    # save as txt
    with open(os.path.join(path_rst,'train_losses.txt'), 'w') as f:
        for v in train_losses:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'train_accuracy.txt'), 'w') as f:
        for v in train_accuracy:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'val_losses.txt'), 'w') as f:
        for v in val_losses:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'val_accuracy.txt'), 'w') as f:
        for v in val_accuracy:
            f.write(str(v)+'\n')

    with open(os.path.join(path_rst,'best_val_accuracy.txt'), 'w') as f:
        f.write(str(bestacc))