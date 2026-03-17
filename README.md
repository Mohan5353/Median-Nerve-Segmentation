# Median Nerve Segmentation - Flash Attention V4 (Blackwell Optimized)

This branch contains the **Flash Attention v4** optimized version of the Median Nerve Segmentation project. It is specifically architected for **NVIDIA Blackwell (GB10)** GPUs, utilizing the most advanced attention kernels available.

## Key Features
- **Flash Attention V4 Integration:** Cutting-edge attention implementation using the `flash_attn.cute` interface, specifically optimized for Blackwell's micro-architecture.
- **Robust Fallbacks:** Intelligently falls back to Flash Attention 2 or PyTorch SDPA if V4-specific kernels are unavailable or if complex masks are used.
- **BF16 Precision:** Fully optimized for BFloat16 training, leveraging Blackwell's dedicated hardware for massive throughput improvements.
- **Advanced UI:** Integrated modern `tqdm` progress bars with real-time cumulative metric tracking (Dice Score, Loss).
- **Unified Dataset:** Supports training on the full available dataset using the `--same_dataset` flag.

## Installation

### Prerequisites
- **GPU:** NVIDIA Blackwell (GB10) - *Mandatory for V4 kernels*.
- **CUDA:** 13.0+
- **Environment:** Python 3.11+

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Mohan5353/Median-Nerve-Segmentation.git
cd Median-Nerve-Segmentation
git checkout Flash-Attn-V4

# Create environment
conda create -n vistr_fa4 python=3.11
conda activate vistr_fa4

# Install Core Stack
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

To training the VisTR model with Flash Attention V4:

```bash
cd Median-Nerve-Segmentation/
export PYTHONUNBUFFERED=1
python train.py --model_name vistr --loss bce dice --epochs 100 --batch_size 1 --num_workers 10 --device cuda --data_path ~/DATA-VisTr/ --same_dataset
```

## Optimization Note
This branch includes surgical fixes for the DCN (Deformable Convolution) module to ensure compatibility with BF16 training, forcing the DCN kernels to run in FP32 while keeping the rest of the model in high-speed BF16.

---
## Acknowledgements
Optimization kernels provided by [Flash Attention](https://github.com/Dao-AILab/flash-attention) and specifically the experimental Blackwell interface.
