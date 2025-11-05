import numpy as np
import cv2

import torch
import torchvision
import os

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.special import expit, logit
from scipy.optimize import bisect


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
            self.data_dir = os.path.join(hparams['data_dir'], 'train') 
            self.num_ims = 180000
        else:
            self.data_dir = os.path.join(hparams['data_dir'], 'test') 
            self.num_ims = 18000
        self.size = 270

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

    def __getitem__(self, idx, tk=None):
        if not tk:
            #tk = torch.randint(low=1, high=self.T+1, size=(1,)) # offset by the first 0
            tk = self.sample_tk(dist=self.dist)
        tp = self.observationTimes[tk]
        
        im_path = f'{idx}.png'
        im_label_path = f'{idx}_label.png'
        xT = cv2.imread(os.path.join(self.data_dir, im_path))[:,:,0]
        x0 = cv2.imread(os.path.join(self.data_dir, im_label_path))[:,:,0]

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

        return torch.from_numpy(xt).float(), torch.from_numpy(rates).float(), tk, wk