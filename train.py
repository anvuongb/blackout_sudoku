
import argparse

import torch
import torchinfo
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import lightning as L

from model import SRDModel, LSRDModel
from datasets import BlackoutSudokuDataset
from utils import get_latest_model_pl
import json, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training scale model')
    parser.add_argument('--version', type=str,default='1',help='Version number of the model checkpoints')
    parser.add_argument('--normalize', type=int, default=0, help='if 1, try to normalize the data')
    parser.add_argument('--warmup', type=int, default=0, help='if 1, do warmup for lr in the first 1000 iterations')
    parser.add_argument('--wandb', type=int, default=0, help='if 1, use wandb logger')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index to use')
    parser.add_argument('--data_dir', type=str, help='path to data dir', required=True)
    parser.add_argument('--load_from', type=str, default='null', required=False)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_iters', type=int, default=-1, help="if not -1, will ignore num_epochs, only applicable when lightning=1")
    parser.add_argument('--save_every', type=int, default=-1, help="save every n iterations")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--disable_w', type=int, default=0, help="if 1, disable w in blackout loss")
    parser.add_argument('--rand', type=int, default=0, help="if 1, randomly add some noise")
    parser.add_argument('--rand_flip', type=int, default=0, help="if 1, randomly flip the image")
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--tEnd', type=float, default=21.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.75, help='LR reduce rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='min_lr when using LR ReduceOnPlateau')
    parser.add_argument('--schedule', type=str, default='blackout')
    parser.add_argument('--time_dist', type=str, default='uniform', help='if not uniform, use blackout likelihood weights to sample tk')
    parser.add_argument('--loss', type=str, default='blackout')


    args = parser.parse_args()

    version = args.version
    epochs = args.num_epochs
    batch_size = args.batch_size
    load_path = args.load_from
    lr = args.lr
    sw = None
    sch = args.schedule
    loss_func = args.loss


    # parse config
    hparams = vars(args)
    hparams['lf'] = args.loss
    hparams['in_ch'] = 1 
    hparams['disable_w'] = args.disable_w == 1
    hparams['normalize'] = args.normalize == 1
    hparams['warmup'] = args.warmup == 1
    hparams['wandb'] = args.wandb == 1
    hparams['rand'] = args.rand == 1
    hparams['rand_flip'] = args.rand_flip == 1

    if hparams['tEnd'] == -1: # default tEnd=15 should work for 255
        hparams['tEnd'] = 15
    savepath = f"./models_sudoku_ver{hparams['version']}"
    hparams['savepath'] = savepath

    print("init model and data with config:")
    print(json.dumps(hparams, sort_keys=True, indent=4))

    data_train = BlackoutSudokuDataset(train=True, hparams=hparams)
    data_val = BlackoutSudokuDataset(train=False, hparams=hparams)

    im_size = data_train.size
    train_loader = DataLoader(data_train, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers'])
    val_loader = DataLoader(data_val, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers'], persistent_workers=True)

    # init model
    model = SRDModel(in_ch=hparams['in_ch'])
    torchinfo.summary(model)

    # init log writer
    log_path = f"_models_sudoku-ver{version}"

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # init scheduler
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100], gamma=hparams['gamma'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hparams['gamma'], min_lr=hparams['min_lr'])

    hparams['device'] = 'cpu'
    hparams['device_count'] = -1
    devices_count = torch.cuda.device_count()
    if devices_count > 0:
        hparams['device_count'] = devices_count
        print(f"Using {devices_count} GPUs")
        hparams['device'] = 'cuda'

    # save hparams
    if not os.path.exists(hparams['savepath']):
        try:
            os.makedirs(hparams['savepath'])
        except:
            pass
    with open(os.path.join(hparams['savepath'], 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # use lightning ??
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    if hparams['wandb']:
        logger = L.pytorch.loggers.WandbLogger(save_dir="runs_wandb", project=hparams['model'], name=current_time + "_" + socket.gethostname() + log_path, log_model="all")
    else:
        logger = L.pytorch.loggers.TensorBoardLogger("runs", name=current_time + "_" + socket.gethostname() + log_path)

    # find if ckpt exists
    # note: this only loads the model
    # check out lightning documents to load the whole optimizer + scheduler states
    # seems funky
    if hparams["load_from"] != "null":
        ckpt_path = hparams["load_from"]
    else:
        ckpt_path = get_latest_model_pl(savepath)
    if ckpt_path:
        print(f"Loading model from {ckpt_path}")
        lmodel = LSRDModel.load_from_checkpoint(ckpt_path, model=model, hparams=hparams)
    else:
        lmodel = LSRDModel(model, hparams)

    save_every = None if hparams['save_every'] == -1 else hparams['save_every']
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=savepath, save_top_k=2, save_last=True, every_n_train_steps=save_every, monitor="Loss /train")

    max_epochs = hparams['num_epochs'] if hparams['num_iters'] == -1 else None
    max_steps = hparams['num_iters']
    trainer = L.Trainer(logger=logger, accelerator='auto', devices=-1, gradient_clip_val=1.0, log_every_n_steps=5,
                        max_epochs=max_epochs, max_steps=max_steps,
                        accumulate_grad_batches=1,
                        callbacks=[
                                checkpoint_callback
                        ])
    trainer.fit(lmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)