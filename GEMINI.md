# GEMINI.md - Median Nerve Segmentation Progress (Blackwell Optimized)

## Project Overview
This project focuses on median nerve segmentation using deep learning. We have specifically optimized the VisTR (Video Instance Segmentation Transformer) model for high-performance execution on NVIDIA Blackwell GPUs.

### Environment & Hardware
- **Machine:** Blackwell Workstation (SM 12.0 / GB10).
- **Branch:** `Flash-Attn-V4` (Latest).
- **Conda Env:** `vistr_fa2` (Python 3.11).
- **Core Stack:** PyTorch 2.10.0 (Stable) + CUDA 13.0.
- **Flash Attention:** Version 2.8.3 (Compiled for SM 12.0).
- **DCN Module:** Recompiled from source for CUDA 13.0 architecture.

### Key Modifications & Optimizations
1. **Flash Attention 2 Integration:**
   - Modified `models/VisTR/models/transformer.py`.
   - Switched to the official `flash-attn` library (FA2).
   - Implemented `flash_attn_varlen_func` to handle `key_padding_mask` efficiently by unpadding/padding sequences on the fly, avoiding unnecessary computation on pad tokens.
   - Robust fallback to PyTorch **SDPA** for complex masks or non-CUDA/FP32 paths.

2. **Dataset & Split Logic:**
   - **Structure:** Refactored `dataset.py` to support nested `patient_id/*_IMAGES/images/` and `masks/` directories.
   - **Formats:** Added support for `.tif` mask files.
   - **Split:** Implemented a **Patient-wise 80/10/10 Split** (Train/Val/Test) using a fixed seed (0) to ensure frames from the same patient never leak between sets.

3. **PyTorch 2.10 Compatibility:**
   - Updated all `torch.load` calls to use `weights_only=False` to handle `argparse.Namespace` serialization in newer PyTorch versions.
   - Patched `pycocotools` with updated `ytvos.py` and `ytvoseval.py`.

4. **UI & Dataset Flexibility:**
   - **Progress Bar:** Modernized all training loops (UNet & VisTR) with `tqdm` bars showing real-time cumulative metrics (Loss, DSC).
   - **Unified Dataset Mode:** Added `--same_dataset` flag to use the same data for train/val/test splits.

5. **BF16 Precision:**
   - Implemented `torch.amp.autocast(dtype=torch.bfloat16)` across all training/validation loops (VisTR and UNet).
   - Optimized for Blackwell Tensor Cores and required for activating Flash Attention kernels.

### Training & Execution
- **Entry Point:** `train.py` (VisTR, UNet, TransUNet supported).
- **Dataset Path:** Remote: `~/DATA-VisTr/` (approx. 12GB).
- **Status:** Complete codebase and environment replicated to `mohankumar@10.24.38.16`. Dry runs verified successful.

### Standard Training Command
```bash
python train.py --model_name vistr --loss bce dice --epochs 100 --batch_size 1 --num_workers 10 --device cuda --data_path ~/DATA-VisTr/ --same_dataset
```
