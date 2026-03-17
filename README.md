# Median Nerve Segmentation - Flash Attention 2 Branch

This branch contains the **Flash Attention 2** optimized version of the Median Nerve Segmentation project, specifically tuned for high-performance training on NVIDIA Blackwell and Hopper GPUs.

## Key Features
- **Flash Attention 2 Integration:** Optimized Transformer implementation using `flash_attn_func` for superior speed and memory efficiency.
- **BF16 Precision:** Native support for BFloat16 training, leveraging Blackwell Tensor Cores while maintaining numerical stability.
- **Unified Dataset Mode:** Added `--same_dataset` flag to allow using the entire dataset for all training phases.
- **Improved Progress Monitoring:** Real-time cumulative metrics (Loss, DSC) via modern `tqdm` progress bars.
- **Blackwell Optimized:** Custom DCN module recompiled for CUDA 13.0 and optimized for SM 12.0 architecture.

## Installation

### Prerequisites
- NVIDIA Blackwell (GB10) or Hopper GPU.
- CUDA 13.0+
- Python 3.11+

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Mohan5353/Median-Nerve-Segmentation.git
cd Median-Nerve-Segmentation
git checkout flash-attn-v2

# Create and activate environment
conda create -n vistr_fa2 python=3.11
conda activate vistr_fa2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install tqdm scipy pillow cython
```

### Compile DCN Module
```bash
cd Median-Nerve-Segmentation/models/VisTR/models/dcn
python setup.py install
```

## Training

To train the VisTR model with Flash Attention 2 and BF16:

```bash
cd Median-Nerve-Segmentation/
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
