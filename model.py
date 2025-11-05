import torch
import json
from ema_pytorch import EMA
import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def log_loss_blackout(outputs, targets, w, disable_w=False):
    # finite time version
    if disable_w:
        l = outputs - targets * torch.log(outputs + 1e-8)
    else:
        l = w[:,None,None,None]*(outputs - targets * torch.log(outputs + 1e-8))
    return torch.mean(l)

def calc_loss(outputs, targets, w, loss, disable_w=False):
    if loss == 'l1':
        return torch.nn.L1Loss()(outputs, targets) 
    elif loss == 'l2':
        return torch.nn.MSELoss()(outputs, targets) 
    elif loss == 'blackout':
        return log_loss_blackout(outputs, targets, w, disable_w)

class LSRDModel(L.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model 
        self.xhparams = hparams
        self.xdevice = hparams['device']
        
        self.ema = EMA(
                        self.model,
                        beta=0.9999,            # Exponential moving average factor
                        update_after_step=100,  # Start updating after 100 steps
                        update_every=10,        # Update EMA every 10 steps
                    )

        if hparams['wandb']:
            # tensorboard hangs when save hyperparams?
            self.save_hyperparameters(hparams)

    def on_before_zero_grad(self, optimizer):
        self.ema.update()
    
    def configure_optimizers(self):
        # init optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.xhparams['lr'])
        # init scheduler
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100], gamma=hparams['gamma'])
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.xhparams['gamma'], min_lr=self.xhparams['min_lr'])
        warmup = self.xhparams['warmup']
        def warmup_scheduler(iteration):
            max_iters = 1000
            if warmup and iteration < max_iters:
                return (iteration+1)/max_iters
            return 1.0
        scheduler = LambdaLR(optimizer, warmup_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Loss /train", "interval": "step", "frequency":1, 'reduce_on_plateau': False}}
    
    def training_step(self, batch, batch_idx):
        xt, rt, tk, wk = batch
        outputs = self.model(tk, xt) # don't normalize tk
        loss = calc_loss(outputs, rt, wk, self.xhparams['lf'], self.xhparams['disable_w'])

        self.log('Loss /train', loss, prog_bar=True, sync_dist=True)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=False)

        if self.xhparams['lf'] != 'l1': # also visualize l1 loss
            with torch.no_grad():
                l1_loss = calc_loss(outputs, rt, wk, 'l1', self.xhparams['disable_w'])
                self.log('Loss L1 /train', l1_loss, prog_bar=False, sync_dist=True)
                
        #if self.trainer.is_last_batch:
            #sch = self.lr_schedulers()
            #sch.step(self.trainer.callback_metrics["Loss /train"])
        return loss
    
    def validation_step(self, batch, batch_idx):
        xt, rt, tk, wk = batch
        with torch.no_grad():
            outputs = self.model(tk, xt) # don't normalize tk
            loss = calc_loss(outputs, rt, wk, self.xhparams['lf'], self.xhparams['disable_w'])

            self.log('Loss /val', loss, prog_bar=True, sync_dist=True)

            if self.xhparams['lf'] != 'l1': # also visualize l1 loss
                l1_loss = calc_loss(outputs, rt, wk, 'l1', self.xhparams['disable_w'])
                self.log('Loss L1 /val', l1_loss, prog_bar=False, sync_dist=True)
            return loss

class SRDModel(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        config = {
            'channels': in_ch,
            'dimensions':2,
            'progressive':'none',
            'progressive_input':'residual',
            'combine_method':'sum',
            'init_scale':1e-5, # original ncspp used 1e-10 
            'num_res_blocks':4
        }

        config['ch_mult'] = [1, 2, 2, 2]

        config['nf'] = 128
        config['dropout'] = 0.1
        from score_models import ScoreModel, NCSNpp
        print('model config')
        print(json.dumps(config, sort_keys=True, indent=4))
        net = NCSNpp(**config)

        # Adjust the model
        for name, module in list(net.named_modules()):
            if isinstance(module, torch.nn.Conv2d) and module.out_channels == in_ch:
                new_module = torch.nn.Sequential(module, torch.nn.Softplus())
        
                parent_name, child_name = name.split('.')[:2]
                getattr(net, parent_name)[int(child_name)] = new_module
                getattr(net, parent_name)[int(child_name)].weight = new_module[0].weight

        self.model = ScoreModel(model=net, beta_min=0.1, beta_max=20)
    
    def forward(self, tk, xt):
        return self.model.model(tk, xt) # skipped variance scaling