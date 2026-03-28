import os
import time
import torch
import argparse
import warnings
import datetime
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import get_dataset
from models.models import get_model
import models.VisTR.util.misc as utils

# Suppress all non-critical warnings
warnings.filterwarnings("ignore")

class CooldownCallback(Callback):
    """Custom callback to implement the 5-minute machine cooldown between epochs."""
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < trainer.max_epochs - 1:
            if trainer.is_global_zero:
                print(f"\nCooldown: Sleeping for 5 minutes before Epoch {trainer.current_epoch + 1}...")
            time.sleep(300)

class VisTRLightningModule(LightningModule):
    def __init__(self, args, train_dataset=None, val_dataset=None):
        super().__init__()
        self.args = args
        self.train_set = train_dataset
        self.val_set = val_dataset
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])
        
        # Instantiate model, criterion, and postprocessors
        self.model, self.criterion, self.postprocessors = get_model(args)
        
        # Load pretrained weights if provided
        if args.pretrained_weights:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)['model']
            # Remove keys that don't match for fine-tuning
            for key in ["vistr.class_embed.weight", "vistr.class_embed.bias", "vistr.query_embed.weight"]:
                if key in checkpoint:
                    del checkpoint[key]
            self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, samples):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        self.log("val_loss", losses, on_epoch=True, prog_bar=True, sync_dist=True)
        return losses

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.args.batch_size, 
            collate_fn=utils.collate_fn, 
            num_workers=self.args.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.args.batch_size, 
            collate_fn=utils.collate_fn, 
            num_workers=self.args.num_workers,
            shuffle=False
        )

def get_args_parser():
    from train import get_args_parser as base_parser
    parser = argparse.ArgumentParser('VisTR Lightning Training', parents=[base_parser()])
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--strategy', default='ddp', type=str)
    parser.add_argument('--nodes', default=1, type=int)
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    # Tell get_dataset to return raw datasets
    args.return_dataset = True
    args.distributed = False # Let Lightning handle DDP sampling internally
    
    train_dataset, val_dataset = get_dataset(args)

    model = VisTRLightningModule(args, train_dataset=train_dataset, val_dataset=val_dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        monitor='val_loss'
    )
    
    logger = TensorBoardLogger("lightning_logs", name=args.model_name)

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.world_size if args.device == 'cuda' else 1,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        precision='bf16-mixed',
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10), CooldownCallback()],
        logger=logger,
        log_every_n_steps=10
    )

    print(f"Starting Lightning Training for {args.model_name}...")
    trainer.fit(model)

if __name__ == '__main__':
    main()
