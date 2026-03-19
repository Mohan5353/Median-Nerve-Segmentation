# Median Nerve Segmentation - Flash Attention 2 + DDP Branch

This branch contains the **Flash Attention 2** optimized version of the Median Nerve Segmentation project with **Distributed Data Parallel (DDP)** support for multi-GPU training.

## Key Features
- **Flash Attention 2 Integration:** Optimized Transformer implementation using `flash_attn_func`.
- **Distributed Data Parallel (DDP):** Support for scaling training across multiple GPUs.
- **BF16 Precision:** Native BFloat16 training for maximum performance on Blackwell/Hopper.
- **Torch Compile:** Model compilation enabled for both training and inference.
- **Unified Dataset Mode:** `--same_dataset` flag for full dataset utilization.

## Training with DDP

To train on multiple GPUs (e.g., 2 GPUs), use `torchrun`:

```bash
cd Median-Nerve-Segmentation/
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=2 train.py --model_name vistr --loss bce dice --epochs 100 --batch_size 1 --num_workers 4 --device cuda --data_path ~/DATA-VisTr/ --same_dataset
```

For single GPU training, the standard command still works:
```bash
python train.py --model_name vistr --loss bce dice --epochs 100 --batch_size 1 --num_workers 10 --device cuda --data_path ~/DATA-VisTr/ --same_dataset
```

## Testing
To test the model:
```bash
python test.py --model_name vistr --data_path ~/DATA-VisTr/ --load_from checkpoints_vistr_resnet101/checkpoint.pth
```

---
## Acknowledgements
This project builds upon [VisTR](https://github.com/EpiphanyOfWorld/VisTR) and [DETR](https://github.com/facebookresearch/detr). Optimization kernels provided by [Flash Attention](https://github.com/Dao-AILab/flash-attention).
