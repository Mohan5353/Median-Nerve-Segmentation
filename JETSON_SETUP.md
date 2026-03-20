# Jetson Orin NX - Optimized Inference Setup

This document records the exact steps taken to configure the Jetson Orin NX (ssh `nvidia@10.24.38.17`) for high-performance median nerve segmentation inference.

## 1. System Clock and Repository Fix
Jetson often arrives with an incorrect system date, which causes `apt` and `pip` certificate failures.

```bash
# Fix system time (Crucial for SSL certificates)
sudo date -s "2026-03-19 17:30:00"

# Enable NVIDIA L4T repositories
sudo sed -i 's/#deb/deb/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

# Update system
sudo apt update
```

## 2. NVIDIA Hardware Drivers (JetPack)
Ensure the full NVIDIA stack is installed for CUDA support.

```bash
sudo apt install -y nvidia-jetpack
```

## 3. Global Environment Configuration
Add CUDA to the system path permanently in `~/.bashrc`.

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 4. Python Environment (Conda/Miniforge)
Use Miniforge for native ARM64 (aarch64) support.

```bash
# Create inference environment
conda create -y -n vistr_orin python=3.11
conda activate vistr_orin

# Install base dependencies
pip install numpy scipy pillow cython tqdm ninja packaging
```

## 5. CUDA-Enabled PyTorch
For JetPack 6, we used the specific cu126 index to force the GPU-enabled version.

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

## 6. Flash Attention 2 Compilation
Flash Attention 2 must be compiled from source for the Orin's Ampere architecture (`sm_87`).

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=2 # Low parallelism to prevent OOM
pip install flash-attn --no-build-isolation
```

## 7. DCN Module (In-Place)
Compilation of the Deformable Convolution module.

```bash
cd /home/nvidia/VisTr/Median-Nerve-Segmentation/models/VisTR/models/dcn
rm -rf build/ _C*.so
python setup.py build_ext --inplace

# Create symbolic link for easier importing
ln -sf _C.cpython-311-aarch64-linux-gnu.so _C.so
```

## 8. Verified Inference Command
Run inference using the optimized `flash-attn-v2` branch and `torch.compile`.

```bash
export PYTHONUNBUFFERED=1
python test.py --model_name vistr --data_path /home/nvidia/DATA-VisTr/ --load_from /home/nvidia/VisTr/checkpoints/checkpoint.pth
```
