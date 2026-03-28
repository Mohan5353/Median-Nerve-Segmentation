# Median Nerve Segmentation - Flash Attention 2 + PyTorch Lightning

This branch contains the **PyTorch Lightning** implementation of the project, optimized for **Flash Attention 2** and Multi-GPU training (DDP).

## Key Features
- **PyTorch Lightning Module:** Reorganized code for better modularity and stability.
- **Flash Attention 2:** High-speed attention kernels for Blackwell/Ampere.
- **Native DDP:** Automated multi-GPU synchronization and scaling.
- **Machine Safety:** Custom `CooldownCallback` implements a 5-minute pause between epochs.
- **Automatic Precision:** Managed `bf16-mixed` precision for Blackwell Tensor Cores.

## Training with PyTorch Lightning (Multi-GPU)

To train on multiple GPUs (e.g., 2x A6000) using the Lightning Trainer:

```bash
cd Median-Nerve-Segmentation/
export PYTHONUNBUFFERED=1

python train_lightning.py \
    --model_name vistr \
    --loss bce dice \
    --epochs 100 \
    --batch_size 1 \
    --world_size 2 \
    --accelerator gpu \
    --strategy ddp \
    --data_path /home/vaishnavi/DATA-VisTr/ \
    --same_dataset
```

## Advantages of this Version
1. **Simplified Setup:** No need for manual `init_distributed_mode` or `DistributedSampler` logic.
2. **Robust Logging:** Built-in TensorBoard support (`lightning_logs/`).
3. **Smart Checkpointing:** Automatically saves the best 3 models based on `val_loss`.
4. **Resumability:** Better handling of `trainer.fit(model, ckpt_path=...)`.


## Testing
To test the model:
```bash
python test.py --model_name vistr --data_path ~/DATA-VisTr/ --load_from checkpoints_vistr_resnet101/checkpoint.pth
```

---
## Acknowledgements
This project builds upon [VisTR](https://github.com/EpiphanyOfWorld/VisTR) and [DETR](https://github.com/facebookresearch/detr). Optimization kernels provided by [Flash Attention](https://github.com/Dao-AILab/flash-attention).
