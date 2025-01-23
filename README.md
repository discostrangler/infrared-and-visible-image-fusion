# Evaluating Image Fusion Techniques for Low Light Surveillance

This repository contains the code and results for research conducted on a detailed analysis of the DIDFuse algorithm to enhance low-light surveillance.

## Repository Contents

### Implemented Model:
1. **DIDFuse** - Autoencoder-based image fusion method for low-light scenarios.
<img width="735" alt="Screenshot 2025-01-22 at 3 03 30 PM" src="https://github.com/user-attachments/assets/34017f64-5948-4416-9334-1244fd8ea92c" />

### Dataset:
- **LLVIP Dataset**: This dataset includes paired visible and infrared images under low-light conditions. It contains 16,836 image pairs across 26 diverse scenes captured between 1800 and 2200 hours.
- **Preprocessing**: Images are converted to grayscale and resized to 256x256 for uniformity during training.
  <img width="474" alt="Screenshot 2025-01-22 at 3 02 16 PM" src="https://github.com/user-attachments/assets/c539e861-a167-433b-86dd-230855c10e45" />

### Results:
- Quantitative analysis using metrics like Entropy (EN), Mutual Information (MI), Peak Signal-to-Noise Ratio (PSNR), Visual Information Fidelity (VIF), and more.
- Qualitative results showcasing fused images produced by DIDFuse.
<img width="737" alt="Screenshot 2025-01-22 at 3 02 39 PM" src="https://github.com/user-attachments/assets/f43b046a-fb82-4c73-b60b-5f5a75e00b86" />

### Dependencies:
- Python >= 3.8
- PyTorch >= 1.9
- NumPy, Matplotlib, OpenCV, SimpleITK
- Other requirements are listed in `requirements.txt`.

## Usage

### Data Preparation
- Download the [LLVIP Dataset](https://bupt-ai-cz.github.io/LLVIP/) and place it in the `datasets/LLVIP` folder.

### Training Model
Run the training script for DIDFuse:
```bash
python scripts/train.py --model DIDFuse --dataset datasets/LLVIP
```

### Evaluation
Evaluate the DIDFuse model using:
```bash
python scripts/evaluate.py --model DIDFuse --dataset datasets/LLVIP
```

### Results Visualization
Fused images and metrics are saved in the `results/` folder.


## License
This project is licensed under the MIT License.
