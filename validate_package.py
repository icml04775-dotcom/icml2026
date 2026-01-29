#!/usr/bin/env python3
"""
Validation script for supplementary materials package.
Checks file structure and integrity without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def check_file(path, required=True):
    """Check if a file exists."""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_str = format_size(size)
        print(f"  ✓ {path} ({size_str})")
        return True
    else:
        status = "✗" if required else "⚠"
        print(f"  {status} {path} - {'MISSING' if required else 'optional'}")
        return not required

def format_size(bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"

def count_python_files(directory):
    """Count Python files in a directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        count += sum(1 for f in files if f.endswith('.py'))
    return count

def main():
    print("=" * 60)
    print("Supplementary Materials Package Validation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('sae_explore.py'):
        print("\n❌ Error: Not in supplementary directory!")
        print("Please run this script from the supplementary/ folder.")
        return False
    
    all_ok = True
    
    # Check core files
    print("\n📄 Core Files:")
    all_ok &= check_file('README.md')
    all_ok &= check_file('requirements.txt')
    all_ok &= check_file('sae_explore.py')
    all_ok &= check_file('MANIFEST.md')
    
    # Check model directory
    print("\n🧠 Model Code:")
    all_ok &= check_file('model/__init__.py')
    all_ok &= check_file('model/dinov3_sae_topk_model.py')
    all_ok &= check_file('model/dinov3_model.py')
    all_ok &= check_file('model/dinov3_sae_model.py')
    all_ok &= check_file('model/depth_dpt.py')
    all_ok &= check_file('model/layers.py')
    
    # Count Python files
    py_count = count_python_files('model')
    print(f"\n  📊 Total Python files in model/: {py_count}")
    
    # Check SAE module
    print("\n🔧 SAE Module:")
    all_ok &= check_file('model/sae/__init__.py')
    all_ok &= check_file('model/sae/topk_sparse_autoencoder.py')
    all_ok &= check_file('model/sae/sparse_autoencoder.py')
    all_ok &= check_file('model/sae/sae_loss.py')
    
    # Check DINOv3
    print("\n🦕 DINOv3 Implementation:")
    all_ok &= check_file('model/dinov3/layers/__init__.py')
    all_ok &= check_file('model/dinov3/models/__init__.py')
    all_ok &= check_file('model/dinov3/utils/__init__.py')
    
    # Check DPT layers
    print("\n🔲 DPT Decoder:")
    all_ok &= check_file('model/dpt_layers/blocks.py')
    all_ok &= check_file('model/dpt_layers/transform.py')
    
    # Check weights
    print("\n⚖️  Model Weights:")
    weights_exist = check_file('weights/model_checkpoint.ckpt')
    if weights_exist:
        size = os.path.getsize('weights/model_checkpoint.ckpt')
        if size > 1e9:  # More than 1GB
            print(f"    ℹ️  Large file: Consider hosting separately for paper submission")
    all_ok &= weights_exist
    
    # Check data
    print("\n📊 Sample Data:")
    all_ok &= check_file('data/sample_image.tif')
    
    # Check directories
    print("\n📁 Directories:")
    for dirname in ['outputs']:
        if os.path.isdir(dirname):
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ⚠ {dirname}/ - will be created on first run")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Package validation PASSED")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run notebook: marimo edit sae_explore.py")
        return True
    else:
        print("❌ Package validation FAILED")
        print("\nSome required files are missing. Please check the MANIFEST.md")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
