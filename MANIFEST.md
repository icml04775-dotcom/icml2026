# Supplementary Materials Manifest

## Package Information

**Version**: 1.0  
**Created**: January 2026  
**Total Size**: ~7.5 GB  
**Python Files**: 42

## File Structure Verification

### Core Files
- ✅ `README.md` - Installation and usage instructions
- ✅ `requirements.txt` - Python dependencies
- ✅ `sae_explore.py` - Main exploration notebook (Marimo)
- ✅ `MANIFEST.md` - This file

### Model Code (`model/`)
- ✅ `__init__.py` - Package initialization
- ✅ `dinov3_sae_topk_model.py` - TopK SAE segmentation model (39KB)
- ✅ `dinov3_model.py` - DINOv3 backbone
- ✅ `dinov3_sae_model.py` - SAE mixins
- ✅ `depth_dpt.py` - DPT decoder head
- ✅ `layers.py` - Utility layers

#### SAE Module (`model/sae/`)
- ✅ `__init__.py`
- ✅ `topk_sparse_autoencoder.py` - TopK SAE implementation
- ✅ `sparse_autoencoder.py` - Base SAE
- ✅ `sae_loss.py` - Loss functions
- ✅ `activation_store.py` - Activation caching
- ✅ `sae_training.py` - Training utilities
- ✅ `feature_extractor.py` - Feature extraction

#### DINOv3 Implementation (`model/dinov3/`)
- ✅ `layers/` - ViT layers (attention, FFN, etc.)
- ✅ `models/` - Vision transformer models
- ✅ `utils/` - Utility functions

#### DPT Decoder (`model/dpt_layers/`)
- ✅ `blocks.py` - DPT building blocks
- ✅ `transform.py` - Feature transformation

#### Legacy Layers (`model/dinov2_layers/`)
- ✅ Various ViT layer implementations

### Model Weights (`weights/`)
- ✅ `model_checkpoint.ckpt` - Trained model checkpoint (7.5 GB)
  - Format: PyTorch Lightning checkpoint
  - Contains: Model state dict, optimizer state, training metadata
  - SHA256: [Compute with: `sha256sum model_checkpoint.ckpt`]

### Sample Data (`data/`)
- ✅ `sample_image.tif` - Example aerial image (2.9 MB)
  - Format: GeoTIFF
  - Size: 1024x1024 pixels
  - Channels: 3 (RGB)
  - Location: Austin, TX region

### Output Directory (`outputs/`)
- Directory for generated visualizations and results
- Created automatically on first run

## Verification Commands

### Check file integrity
```bash
# Navigate to supplementary directory
cd supplementary/

# Verify Python files
find model -name "*.py" | wc -l  # Should return: 42

# Verify checkpoint exists
ls -lh weights/model_checkpoint.ckpt  # Should show: 7.5G

# Verify sample data
ls -lh data/sample_image.tif  # Should show: 2.9M

# Check total size
du -sh .  # Should show: ~7.5G
```

### Compute checksums
```bash
# Checkpoint checksum
sha256sum weights/model_checkpoint.ckpt > CHECKSUMS.txt

# Sample data checksum
sha256sum data/sample_image.tif >> CHECKSUMS.txt
```

## Dependencies Summary

**Core Requirements**:
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 16GB+ GPU RAM (recommended)

**Full list**: See `requirements.txt`

## Quick Validation

Run this to verify the package structure:

```bash
python3 -c "
import os
import sys

required_files = [
    'README.md',
    'requirements.txt',
    'sae_explore.py',
    'model/__init__.py',
    'model/dinov3_sae_topk_model.py',
    'weights/model_checkpoint.ckpt',
    'data/sample_image.tif',
]

missing = []
for f in required_files:
    if not os.path.exists(f):
        missing.append(f)

if missing:
    print('❌ Missing files:')
    for f in missing:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('✅ All required files present')
"
```

## Known Issues

1. **Large File Size**: The checkpoint file is 7.5GB. For paper submission:
   - Host on Zenodo, Figshare, or institutional repository
   - Provide download link in README
   - Include checksum for verification

2. **GPU Requirements**: Model requires significant GPU memory
   - Reduce `img_size` if OOM errors occur
   - CPU inference is possible but slow

3. **CUDA Compatibility**: Ensure PyTorch CUDA version matches system CUDA
   - Check with: `nvidia-smi`
   - Reinstall PyTorch if needed

## Usage Notes

- First-time setup requires installing dependencies (see README.md)
- Marimo must be installed to run the interactive notebook
- Internet connection required for first run (downloads pretrained backbone weights if not in checkpoint)

## Support

For issues or questions:
- Check README.md for troubleshooting
- Verify all files using checksums
- Ensure dependencies are correctly installed

---

**Generated**: January 29, 2026  
**Package Version**: 1.0
