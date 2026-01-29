# Supplementary Materials: SAE Feature Exploration for Aerial Image Segmentation

## Overview

This supplementary package contains code for exploring learned features in a Sparse Autoencoder (SAE) integrated with a DINOv3-based segmentation model for aerial imagery. The interactive notebook demonstrates feature visualization, class-specific activation patterns, and feature steering capabilities.

## Package Contents

```
supplementary/
├── README.md                      # This file
├── requirements.txt                # Python dependencies
├── sae_explore.py                 # Main exploration notebook (Marimo)
├── model/                         # Model implementation
│   ├── dinov3_sae_topk_model.py  # TopK SAE segmentation model
│   ├── dinov3_model.py            # DINOv3 backbone
│   ├── dinov3_sae_model.py       # SAE mixins
│   ├── depth_dpt.py               # DPT decoder head
│   ├── layers.py                  # Utility layers
│   ├── sae/                       # SAE implementation
│   ├── dinov3/                    # DINOv3 layers & models
│   └── dpt_layers/                # DPT decoder layers
├── weights/                       # Model checkpoints
│   └── model_checkpoint.ckpt     # Trained model (7.5GB)
├── data/                          # Sample data
│   └── sample_image.tif          # Example aerial image
└── outputs/                       # Output directory (created on run)
```

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **GPU**: CUDA-capable GPU with at least 16GB VRAM (recommended)
  - CPU inference is possible but significantly slower
- **CUDA Toolkit**: 11.8 or higher (for GPU support)
- **Storage**: At least 10GB free space

### Setup Steps

1. **Create a Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PyTorch GPU support** (if using GPU):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

   If CUDA is not available, install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Running the Interactive Notebook

The main exploration notebook is implemented using [Marimo](https://marimo.io/), an interactive Python notebook framework.

**Start the notebook**:
```bash
marimo edit sae_explore.py
```

This will open an interactive web interface in your browser where you can:
- Visualize model predictions on the sample image
- Explore sparse feature activations for different land cover classes
- Compute class-specific feature profiles
- Perform feature steering experiments
- Propagate edits to similar regions globally

### Running Specific Sections

If you prefer to run specific parts of the analysis as a script:
```bash
marimo run sae_explore.py
```

### Key Notebook Sections

1. **Model Loading**: Loads the trained segmentation model with SAE
2. **Feature Extraction**: Extracts sparse features from the backbone
3. **Class Profile Analysis**: Computes class-specific feature activation patterns
4. **Feature Visualization**: Visualizes which features activate for different classes
5. **Feature Steering**: 
   - Local steering: Modify activations in selected regions
   - Global steering: Propagate changes to similar regions
6. **Decoder Manipulation**: Directly modify SAE decoder weights

## Key Capabilities

### 1. Feature Visualization
Examine which sparse features activate for different land cover classes (e.g., trees, roads, buildings).

### 2. Class Profile Analysis
Compute aggregated feature activation patterns for each class to understand class-specific representations.

### 3. Feature Steering
- **Boost**: Increase activation of specific features
- **Suppress**: Zero out specific features
- **Gap-filling**: Add missing class features to change predictions

### 4. Global Propagation
Find all image patches similar to a selected region and apply the same steering intervention.

## Model Architecture

- **Backbone**: DINOv3-Large (ViT-L/14) pre-trained on aerial imagery
- **SAE**: TopK Sparse Autoencoder with 65,536 dictionary features
  - Guaranteed K=32 active features per token
  - Unified dictionary across 4 backbone layers
- **Decoder**: Dense Prediction Transformer (DPT) head
- **Classes**: 9 land cover classes (bareland, rangeland, tree, water, etc.)

## Configuration

Key parameters can be modified in the configuration cell of `sae_explore.py`:

```python
sae_hidden_dim = 65536   # SAE dictionary size
num_classes = 9          # Number of segmentation classes
img_size = 1024          # Input image size
patch_size = 16          # ViT patch size
device = "cuda:0"        # Device (use "cpu" for CPU inference)
```

## Model Weights

**Size**: 7.5 GB  
**Format**: PyTorch Lightning checkpoint (`.ckpt`)

### Downloading the Model

Due to the large file size, the trained model checkpoint is hosted separately on Google Drive:

**Download Link**: [https://drive.google.com/drive/folders/1i4p00D9TnOWFr99ujOJp4JW7h_qNYD-D?usp=sharing](https://drive.google.com/drive/folders/1i4p00D9TnOWFr99ujOJp4JW7h_qNYD-D?usp=sharing)

### Installation Steps

1. Download the `model_checkpoint.ckpt` file from the Google Drive link above
2. Place the downloaded file in the `weights/` directory:
   ```bash
   # Create the weights directory if it doesn't exist
   mkdir -p weights/
   
   # Move the downloaded checkpoint to the correct location
   mv ~/Downloads/model_checkpoint.ckpt weights/
   ```
3. Verify the file is in the correct location:
   ```bash
   ls -lh weights/model_checkpoint.ckpt
   # Should show: weights/model_checkpoint.ckpt (~7.5 GB)
   ```

The notebook will automatically load the model from `weights/model_checkpoint.ckpt`.

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `img_size` to 512 or 256
- Use CPU inference: Change `device = "cpu"` (slower)
- Close other GPU applications

### ImportError: Module not found
- Ensure you're in the supplementary directory when running
- Verify all dependencies are installed: `pip list`

### CUDA/GPU Issues
- Check CUDA installation: `nvidia-smi`
- Reinstall PyTorch with matching CUDA version
- Use CPU inference as fallback

### Marimo not opening
- Check if port 8080 is available
- Try: `marimo edit sae_explore.py --port 8081`

---

**Last Updated**: January 2026  
**Tested on**: Ubuntu 20.04, Python 3.10, PyTorch 2.1.0, CUDA 11.8
