import numpy as np
import cv2

import torch
import torchvision
import os

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.special import expit, logit
from scipy.optimize import bisect
import pandas as pd

import torch

def f(x):
    return np.log(x/(1-x))

def get_schedule(tEnd, T, sch='linear'):
    if sch == 'linear':
        return np.linspace(0, tEnd, T)
    elif sch == 'blackout':
        #k = np.arange(0, T)
        #tp =  - np.log(expit(logit(1-np.exp(-tEnd)) + (k-1)/(T-1)*(logit(np.exp(-tEnd))-logit(1-np.exp(-tEnd)))))
        # return np.hstack([0, tp])

        xEnd = np.exp(-tEnd)
        fGrid = np.linspace(-f(xEnd), f(xEnd), T)
        xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(T)])
        observationTimes = -np.log(xGrid)
        return np.hstack([0, observationTimes]) # add 0 in the beginning just to fuck with the indices

class BaseDataset(Dataset):
    def __init__(self, train, hparams):
        # construct times
        self.tEnd = hparams['tEnd']
        self.T = hparams['T']
        self.normalize = hparams['normalize']
        self.observationTimes = get_schedule(hparams['tEnd'], hparams['T'], hparams['schedule'])
        self.weights = np.exp(-self.observationTimes[:-1]) - np.exp(-self.observationTimes[1:]) # precompute weights, probably save some vram
        self.dist = hparams['time_dist']

        if train:
            self.data_csv = os.path.join(hparams['data_dir'], 'train.csv') 
            pdf = pd.read_csv(self.data_csv, names=['data','solve'])
            self.data = pdf.values
            self.num_ims = len(pdf)
        else:
            self.data_csv = os.path.join(hparams['data_dir'], 'test.csv') 
            pdf = pd.read_csv(self.data_csv, names=['data','solve'])
            self.data = pdf.values
            self.num_ims = len(pdf)
        self.size = 288

        # mnist digits for image construction
        import collections
        if train:
            self.mnist_len = 5000
            mnist_path = os.path.join(hparams['mnist_dir'], 'training')
        else:
            self.mnist_len = 800
            mnist_path = os.path.join(hparams['mnist_dir'], 'testing')
        classes = np.arange(10)
        self.mnist_dict = collections.defaultdict(list)

        for c in classes:
            cpath = os.path.join(mnist_path, str(c))
            im_paths = os.listdir(cpath)
            im_paths = [os.path.join(cpath, p) for p in im_paths if p.endswith('.png')]
            for p in im_paths:
                self.mnist_dict[c].append(cv2.imread(p)[:,:,0])

    def __len__(self):
        return self.num_ims

    def sample_tk(self, dist='uniform'):
        if dist == 'uniform':
            tk = np.random.randint(1, self.T+1) # off by one since 0 idx is time 0
        else:
            u = np.random.rand()
            tk = np.searchsorted(self.cdf, u)
            while tk == 0 or tk > self.T: # happens with 0 prob but let make sure no out of bound happens
                u = np.random.rand()
                tk = np.searchsorted(self.cdf, u)
        return torch.tensor(tk).reshape((1,))

class BlackoutSudokuDataset(BaseDataset):
    def __init__(self, train, hparams):
        super().__init__(train, hparams)
    
    def make_border(self, im, size, val=255):
        imb = np.ones((size,size))*val
        imb[2:30, 2:30] = im
        return imb

    def __getitem__(self, idx, tk=None):
        if not tk:
            #tk = torch.randint(low=1, high=self.T+1, size=(1,)) # offset by the first 0
            tk = self.sample_tk(dist=self.dist)
        tp = self.observationTimes[tk]
        
        rand_indices = np.random.randint(0, self.mnist_len, 81) # draw from mnist digit
        size = 32 # make the final image to have size 288x288 -> works with 4 layers Unet
        xT = np.zeros((size*9, size*9))
        x0 = np.zeros((size*9, size*9))
        mask = np.zeros((size*9, size*9))
        d, l = self.data[idx]
        for r in range(9):
            for c in range(9):
                digitd = int(d[c+9*r])
                mnist_digitd = self.make_border(self.mnist_dict[digitd][rand_indices[c+9*r]].copy(), size)
                digitl = int(l[c+9*r])
                mnist_digitl = self.make_border(self.mnist_dict[digitl][rand_indices[c+9*r]].copy(), size)

                if digitd == 0:
                    mnist_digitd = self.make_border(np.zeros((28,28)), size)
                    mask[r*size:(r+1)*size, c*size:(c+1)*size] = self.make_border(np.ones((28,28)), size, 0)
                xT[r*size:(r+1)*size, c*size:(c+1)*size] = mnist_digitd
                x0[r*size:(r+1)*size, c*size:(c+1)*size] = mnist_digitl
        x0 = x0.astype(int)
        xT = xT.astype(int)

        # sample pixels
        probs = 1-np.exp(-tp)
        diff = x0-xT
        rates = np.random.binomial(diff, probs)
        xt = x0 - rates

        # add a dummy dimnesion
        rates = rates[None, :, :]
        xt = xt[None, :, :]

        if self.normalize:
            mean_v = (255.0/2*probs)
            xt = xt.astype(np.float64) - mean_v # normalize as in blackout
        wk = self.weights[tk-1]

        return torch.from_numpy(xt).float(), torch.from_numpy(rates).float(), tk, wk, torch.from_numpy(mask)