import os
import time
import torch
import argparse
import warnings
import datetime
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
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
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
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
        
        # Log training losses
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
        # Setup optimizer with different learning rates for backbone
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

def get_args_parser():
    # Reuse your existing parser logic
    from train import get_args_parser as base_parser
    parser = argparse.ArgumentParser('VisTR Lightning Training', parents=[base_parser()])
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--strategy', default='ddp_find_unused_parameters_true', type=str)
    parser.add_argument('--nodes', default=1, type=int)
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Initialize Distributed Mode to set args.distributed for dataset.py
    utils.init_distributed_mode(args)
    
    # Setup DataLoaders
    train_loader, val_loader = get_dataset(args)

    # Instantiate Model
    model = VisTRLightningModule(args)

    # Setup Callbacks and Loggers
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        monitor='val_loss'
    )
    
    logger = TensorBoardLogger("lightning_logs", name=args.model_name)

    # Setup Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.world_size if args.device == 'cuda' else 1,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        precision='bf16-mixed', # Native Blackwell/Ampere BF16 support
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10), CooldownCallback()],
        logger=logger,
        log_every_n_steps=10
    )

    # Start Training
    print(f"Starting Lightning Training for {args.model_name}...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
