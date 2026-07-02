# S2AD: Statistical Spiking Neural Network for Training-Free Image Anomaly Detection via Firing Rate Deviation

**Authors:** Nguyen Minh Tri, Huynh Cong Viet Ngu

## 📖 Overview
This repository contains the official implementation of **S2AD**, a completely training-free Spiking Neural Network (SNN) framework for Image Anomaly Detection (IAD). 

Unlike previous approaches, S2AD eliminates the need for image-generating decoder models or memory banks, performing anomaly detection directly in the latent spiking feature space.

## ✨ Highlights
- **Non-generative Architecture**: S2AD completely eliminates image reconstruction or memory banks. The anomaly detection process is performed directly within the latent spiking feature space.
- **Z-score Distribution Based Evaluation**: Anomaly scores are determined by measuring the firing-rate deviation of samples compared to normal calibration samples using a Z-score distribution.
- **Neuromorphic Hardware Compatibility**: The framework maps Z-score normalization and absolute deviation calculations to fixed synaptic transformations, making it natively compatible and highly efficient on neuromorphic hardware.

## 🧠 Methodology
The S2AD system is built upon the following core components:
1. **SNN Encoder**: Converts a pretrained ANN into an SNN using threshold balancing. Post-conversion, all synaptic weights are frozen and require no further updating (training-free).
2. **Offline Statistical Calibration**: Performs a single forward pass of normal images through the SNN over a time window $T$. This computes the mean and standard deviation of the firing rate at each spatial position.
3. **Synaptic Detection Head**: Computes anomaly scores using a *Soft Z-score* formula. This is implemented as a dual neural network (excitatory - inhibitory) utilizing the ReLU function to calculate absolute deviations directly on hardware.
4. **Synaptic Interpolator**: Replaces traditional, computationally expensive bilinear interpolation with a pre-calculated sparse synaptic routing structure.
5. **MAD Weighting (Mean Absolute Deviation)**: Employs reliability-based synaptic weights to optimally combine multi-scale features. It assigns higher importance to layers with stable signals while suppressing noise from shallow layers.

## 📊 Performance & Efficiency
S2AD achieves state-of-the-art results among SNN-based anomaly detection methods while offering unprecedented energy efficiency:

- **MVTec-AD**: Achieves **86.7% Image AUROC** and **95.6% Pixel AUROC** using a VGG16 backbone, significantly outperforming previous SNN generative models (e.g., FSVAE at 63.0% Image AUC).
- **VisA**: Achieves **85.5% Image AUROC** and **96.1% Pixel AUROC**.
- **Peak Energy Efficiency**: At $T=32$ with VGG16, S2AD consumes merely **1464 µJ** per inference. This makes S2AD **178× more energy-efficient** than ANN PatchCore and over **1700× more efficient** than PaDiM.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Evaluations
To run S2AD on the MVTec-AD dataset:
```bash
python run_all_mvtec.py
```

To run S2AD on the VisA dataset:
```bash
python run_all_visa.py
```

To run the main S2AD script:
```bash
python main_s2ad.py
```

*(Note: Please ensure that the MVTec-AD and VisA datasets are downloaded and properly placed in your data directory before running the scripts.)*
