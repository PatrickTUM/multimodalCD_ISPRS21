# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

# Data
from data_loader import multiCD

# Models
from aux.fully_convolutional_change_detection.unet import Unet
from aux.fully_convolutional_change_detection.siamunet_conc import SiamUnet_conc, SiamUnet_conc_multi
from aux.fully_convolutional_change_detection.siamunet_diff import SiamUnet_diff
from aux.fully_convolutional_change_detection.fresunet import FresUNet

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
import time
from itertools import chain
import time
import warnings
from pprint import pprint

# Global Variables' Definitions
PATH_TO_DATASET = "/mnt/data/ONERA_s1_s2/"
FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure
BATCH_SIZE = 32  # number of elements in a batch
NUM_THREADS = 6  # number of parallel threads in data loader
NET = 'SiamUnet_conc'  # 'Unet', 'SiamUnet_conc-simple', 'SiamUnet_conc', 'SiamUnet_diff', 'FresUNet'
N_EPOCHS = 50  # number of epochs to train the network
TYPE = 4  #  type of input to the network: 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1
LOAD_TRAINED = False  # whether to load a pre-trained model or train the network
DATA_AUG = True  # whether to apply data augmentation (mirroring, rotating) or not
ONERA_PATCHES = True  # whether to train on patches sliced on the original Onera images or not
NORMALISE_IMGS = True  # z-standardizing on full-image basis, note: only implemented for online slicing!
PRE_SLICED = False  # whether to use pre-sliced patches (processed offline) for training or do online-slicing instead

L = 1024
N = 2

# not applicable for our data loader
# PATCH_SIDE = 96
# TRAIN_STRIDE = int(PATCH_SIDE/2) - 1

augm_str  = 'Augm' if DATA_AUG else 'noAugm'
save_path = os.path.join('/mnt/results/multimodal_CD_ISPRS', 'results', f'{NET}-type-{TYPE}-epochs-{N_EPOCHS}-{augm_str}')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, 'plots'))
    os.makedirs(os.path.join(save_path, 'checkpoints'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match IDs of nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # set only passed devices visible


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(net, train_loader, train_dataset, test_dataset, n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_change_accuracy = 0 * t
    epoch_train_nochange_accuracy = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t

    # mean_acc = 0
    # best_mean_acc = 0
    fm = 0
    best_fm = 0

    lss = 1000
    best_lss = 1000

    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    # train the network for a given number of epochs
    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        tot_count = 0
        tot_loss = 0
        tot_accurate = 0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))

        # iterate over batch
        for batch in train_loader:
            # inserted multi-modality here
            S2_1 = Variable(batch['time_1']['S2'].float().cuda())
            S2_2 = Variable(batch['time_2']['S2'].float().cuda())
            if TYPE in [4, 5]:
                S1_1 = Variable(batch['time_1']['S1'].float().cuda())
                S1_2 = Variable(batch['time_2']['S1'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))

            # get predictions, compute losses and optimize network
            optimizer.zero_grad()
            # outputs of the network are [N x 2 x H x W]
            # label is of shape [32, 96, 96]
            if TYPE in [4, 5]:
                output = net(S2_1, S2_2, S1_1, S1_2)
            else:
                output = net(S2_1, S2_2)
            loss = criterion(output, label.long()) 
            loss.backward()
            optimizer.step()

        # step in lr scheduler
        scheduler.step()

        # evaluate network statistics on train split and keep track
        epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec = test(train_dataset)
        epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_train_change_accuracy[epoch_index] = cl_acc[1]
        epoch_train_precision[epoch_index] = pr_rec[0]
        epoch_train_recall[epoch_index] = pr_rec[1]
        epoch_train_Fmeasure[epoch_index] = pr_rec[2]

        # evaluate network statistics on test split and keep track
        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_dataset)
        epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_test_change_accuracy[epoch_index] = cl_acc[1]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]

        print(f'Test F1 in epoch {epoch_index}: {epoch_test_Fmeasure[epoch_index]}')
        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1], label='Train loss')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        plt.legend(handles=[l1_1, l1_2])
        plt.grid()
        plt.gcf().gca().set_xlim(left=0)
        plt.title('Loss')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1], label='Train accuracy')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        plt.legend(handles=[l2_1, l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        plt.title('Accuracy')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=3)
        plt.clf()
        l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_nochange_accuracy[:epoch_index + 1],
                         label='Train accuracy: no change')
        l3_2, = plt.plot(t[:epoch_index + 1], epoch_train_change_accuracy[:epoch_index + 1],
                         label='Train accuracy: change')
        l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1],
                         label='Test accuracy: no change')
        l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1],
                         label='Test accuracy: change')
        plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        plt.title('Accuracy per class')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=4)
        plt.clf()
        l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Train precision')
        l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Train recall')
        l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Train Dice/F1')
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
        plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        plt.title('Precision, Recall and F-measure')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        fm = epoch_train_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            save_str = os.path.join(save_path, 'checkpoints', 'net-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(fm) + '.pth.tar')
            torch.save(net.state_dict(), save_str)

        lss = epoch_train_loss[epoch_index]
        if lss < best_lss:
            best_lss = lss
            save_str = os.path.join(save_path, 'checkpoints', 'net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(lss) + '.pth.tar')
            torch.save(net.state_dict(), save_str)

        if save:
            im_format = 'png'
            plt.figure(num=1)
            plt.savefig(os.path.join(save_path, 'plots', net_name + '-01-loss.' + im_format))
            plt.figure(num=2)
            plt.savefig(os.path.join(save_path, 'plots', net_name + '-02-accuracy.' + im_format))
            plt.figure(num=3)
            plt.savefig(os.path.join(save_path, 'plots', net_name + '-03-accuracy-per-class.' + im_format))
            plt.figure(num=4)
            plt.savefig(os.path.join(save_path, 'plots', net_name + '-04-prec-rec-fmeas.' + im_format))

    out = {'train_loss': epoch_train_loss[-1],
           'train_accuracy': epoch_train_accuracy[-1],
           'train_nochange_accuracy': epoch_train_nochange_accuracy[-1],
           'train_change_accuracy': epoch_train_change_accuracy[-1],
           'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1],
           'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
           'test_change_accuracy': epoch_test_change_accuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    print(pr_rec)

    return out

# run network on full-scene ROI and evaluate performance
def test(dset):
    net.eval()
    tot_loss = 0
    tot_count = 0

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # iterate over all ROI, load modalities and
    for img_index in dset.names:
        print(f"Testing for ROI {img_index}")
        # inserted multi-modality here
        full_imgs = dset.get_img(img_index)
        S2_1_full, S2_2_full, cm_full = full_imgs['time_1']['S2'], full_imgs['time_2']['S2'], full_imgs['label']
        s = cm_full.shape

        if TYPE in [4, 5]:
            S1_1_full, S1_2_full = full_imgs['time_1']['S1'], full_imgs['time_2']['S1']

        steps0 = np.arange(0, s[0], ceil(s[0] / N))
        steps1 = np.arange(0, s[1], ceil(s[1] / N))
        for ii in range(N):
            for jj in range(N):
                xmin = steps0[ii]
                if ii == N - 1:
                    xmax = s[0]
                else:
                    xmax = steps0[ii + 1]
                ymin = jj
                if jj == N - 1:
                    ymax = s[1]
                else:
                    ymax = steps1[jj + 1]
                # inserted multi-modality here
                S2_1 = S2_1_full[:, xmin:xmax, ymin:ymax]
                S2_2 = S2_2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                S2_1 = Variable(torch.unsqueeze(S2_1, 0).float()).cuda()
                S2_2 = Variable(torch.unsqueeze(S2_2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).cuda()

                if TYPE in [4, 5]:
                    S1_1 = S1_1_full[:, xmin:xmax, ymin:ymax]
                    S1_2 = S1_2_full[:, xmin:xmax, ymin:ymax]
                    S1_1 = Variable(torch.unsqueeze(S1_1, 0).float()).cuda()
                    S1_2 = Variable(torch.unsqueeze(S1_2, 0).float()).cuda()

                # predict output via network and compute losses
                # outputs of the network are [N x 2 x H x W]
                if TYPE in [4, 5]:
                    output = net(S2_1, S2_2, S1_1, S1_2)
                else:
                    output = net(S2_1, S2_2)
                loss         = criterion(output, cm.long())
                tot_loss    += loss.data * np.prod(cm.size())
                tot_count   += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                # compare predictions with change maps and count correct predictions
                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                # evaluate TP, TN, FP & FN
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()

    # compute goodness of predictions
    net_loss = tot_loss / tot_count
    net_accuracy = 100 * (tp + tn) / tot_count

    for i in range(n):  # compute classwise accuracies
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)

    # get precision, recall etc
    prec    = tp / (tp + fp)
    rec     = tp / (tp + fn)
    f_meas  = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc  = tn / (tn + fp)
    pr_rec  = [prec, rec, f_meas, prec_nc, rec_nc]

    return net_loss, net_accuracy, class_accuracy, pr_rec

# run predictions for a given network and data set
# and save all predictions as png
def save_test_results(net, dset):
    for name in tqdm(dset.names):
        print(f"Saving prediction on ROI {name}")
        with warnings.catch_warnings():
            full_imgs = dset.get_img(name)
            # inserted multi-modality here
            S2_1, S2_2, cm = full_imgs['time_1']['S2'], full_imgs['time_2']['S2'], full_imgs['label']
            S2_1 = Variable(torch.unsqueeze(S2_1, 0).float()).cuda()
            S2_2 = Variable(torch.unsqueeze(S2_2, 0).float()).cuda()
            if TYPE in [4, 5]:
                S1_1 = full_imgs['time_1']['S1']
                S1_2 = full_imgs['time_2']['S1']
                S1_1 = Variable(torch.unsqueeze(S1_1, 0).float()).cuda()
                S1_2 = Variable(torch.unsqueeze(S1_2, 0).float()).cuda()
                out = net(S2_1, S2_2, S1_1, S1_2)
            else:
                out = net(S2_1, S2_2)
            _, predicted = torch.max(out.data, 1)
            # save plot of difference maps
            I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
            io.imsave(os.path.join(save_path, 'plots', f'{net_name}-{name}.png'), I)


if __name__ == '__main__':

    # initialize data set instances for train and test splits
    data_loader_train = multiCD(PATH_TO_DATASET, split="train", transform=DATA_AUG, run_on_onera_patches=ONERA_PATCHES, use_pre_sliced=PRE_SLICED, normalize=NORMALISE_IMGS)
    train_loader = torch.utils.data.DataLoader(data_loader_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREADS)
    data_loader_test = multiCD(PATH_TO_DATASET, split="test", run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)
    # note: test loader is never used, testing is always done on full-scene images (not on the patches)
    # test_loader = torch.utils.data.DataLoader(data_loader_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)

    # get train split weighting of pixel labels
    weights = torch.FloatTensor(data_loader_train.weights).cuda()
    print(f"Train data set weighting is: {weights}")
    print(f"Total pixel numbers: {data_loader_train.n_pix}")
    print(f"Changed pixel numbers: {data_loader_train.true_pix}")
    print(f"Change-to-total ratio: {data_loader_train.true_pix / data_loader_train.n_pix}")

    # 0-RGB | 1-RGBIr | 2-All bands s.t. resolution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1

    if TYPE == 0:
        if NET == 'Unet': net, net_name = Unet(2*3, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*3, 2), 'FresUNet'
    elif TYPE == 1:
        if NET == 'Unet': net, net_name = Unet(2*4, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*4, 2), 'FresUNet'
    elif TYPE == 2:
        if NET == 'Unet': net, net_name = Unet(2*10, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*10, 2), 'FresUNet'
    elif TYPE == 3:
        if NET == 'Unet': net, net_name = Unet(2*13, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*13, 2), 'FresUNet'
    elif TYPE == 4:
        if NET == 'Unet': net, net_name = Unet(2*13+2*2, 2), 'FC-EF-multi' # same architecture as the other network
        if NET == 'SiamUnet_conc-simple': net, net_name = SiamUnet_conc(13+2, 2), 'FC-Siam-conc-simple'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc_multi((13, 2), 2), 'FC-Siam-conc-complex'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff_multi(13, 2, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet_multi(2 * 13 + 2*13, 2), 'FresUNet'
    elif TYPE == 5:
        if NET == 'Unet': net, net_name = Unet(2*3+2*2, 2), 'FC-EF-multi'  # same architecture as the other network
        if NET == 'SiamUnet_conc-simple': net, net_name = SiamUnet_conc(3+2, 2), 'FC-Siam-conc-simple'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc_multi((3, 2), 2), 'FC-Siam-conc-complex'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff_multi(3, 2, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet_multi(2 * 3 + 2*3, 2), 'FresUNet'
    net.cuda()

    # define loss: logsoftmax output
    criterion = nn.NLLLoss(weight=weights)
    print('Number of trainable parameters:', count_parameters(net))

    # either load a pre-trained model or train a model from scratch
    if LOAD_TRAINED:
        # load e.g. net.load_state_dict(torch.load('net-best_epoch-1_fm-0.7394933126157746.pth.tar'))
        net.load_state_dict(torch.load(os.path.join(save_path, 'checkpoints', 'net_final.pth.tar')))
        print('LOAD OK')
    else:
        t_start = time.time()
        # train the network and, at the end of each epoch,
        # get its performance on train & test split
        out_dic = train(net, train_loader, data_loader_train, data_loader_test)
        t_end = time.time()
        print(out_dic)
        print('Elapsed time:')
        print(t_end - t_start)
        torch.save(net.state_dict(), os.path.join(save_path, 'checkpoints', 'net_final.pth.tar'))
        print('SAVE OK')

    t_start = time.time()
    save_test_results(net, data_loader_test)
    t_end = time.time()
    print('Elapsed time: {}'.format(t_end - t_start))
