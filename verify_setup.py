#!/usr/bin/env python3
"""
Abdominal Emergency AI Segmentation - Environment Verification Script
======================================================================

This script verifies that all required packages are installed correctly
and that the hardware (NVIDIA A6000 GPUs) is accessible for training.

Run this after setting up the conda environment to ensure everything
is configured properly for Phases 1-3 of development.

Usage:
    conda activate abdomen_scanner
    python verify_setup.py
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(item, status, details=""):
    """Print status for an item."""
    status_symbol = "✓" if status else "✗"
    status_color = "\033[92m" if status else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"{status_color}{status_symbol}{reset_color} {item}", end="")
    if details:
        print(f" ({details})", end="")
    print()


def verify_python_version():
    """Verify Python version is 3.10.x."""
    print_header("Python Version Check")
    
    version = sys.version_info
    is_correct = version.major == 3 and version.minor == 10
    
    print_status(
        "Python 3.10",
        is_correct,
        f"found {version.major}.{version.minor}.{version.micro}"
    )
    
    return is_correct


def verify_core_packages():
    """Verify all core Python packages are installed."""
    print_header("Core Package Installation")
    
    packages = {
        # Phase 1: Data Processing
        "pandas": "Data manipulation (CSV parsing)",
        "numpy": "Numerical operations",
        "pydicom": "DICOM file reading",
        "SimpleITK": "Medical image I/O (NIfTI conversion)",
        "nibabel": "NIfTI file handling",
        
        # Phase 2: MedSAM & Image Processing
        "cv2": "OpenCV (image operations)",
        "PIL": "Pillow (image processing)",
        "skimage": "scikit-image (transformations)",
        "sklearn": "scikit-learn (data splitting)",
        
        # Phase 3: Deep Learning & MONAI
        "torch": "PyTorch (deep learning)",
        "torchvision": "PyTorch vision utilities",
        "monai": "MONAI (medical AI framework)",
        "torchmetrics": "Training metrics",
        
        # Utilities
        "matplotlib": "Visualization",
        "tqdm": "Progress bars",
        "yaml": "Configuration files",
    }
    
    all_installed = True
    
    for package, description in packages.items():
        try:
            __import__(package)
            print_status(f"{package:20s}", True, description)
        except ImportError:
            print_status(f"{package:20s}", False, f"MISSING - {description}")
            all_installed = False
    
    return all_installed


def verify_pytorch_cuda():
    """Verify PyTorch can access CUDA GPUs."""
    print_header("PyTorch & CUDA Configuration")
    
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available)
        
        if cuda_available:
            # GPU count
            gpu_count = torch.cuda.device_count()
            print_status(
                "GPU Count",
                gpu_count >= 2,
                f"{gpu_count} GPU(s) detected (need 2x A6000)"
            )
            
            # GPU details
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # CUDA version
            cuda_version = torch.version.cuda
            print_status("CUDA Version", True, f"v{cuda_version}")
            
            # PyTorch version
            pytorch_version = torch.__version__
            print_status("PyTorch Version", True, f"v{pytorch_version}")
            
            return cuda_available and gpu_count >= 2
        else:
            print("\n⚠️  WARNING: No CUDA GPUs detected!")
            print("   - Check NVIDIA driver installation")
            print("   - Verify CUDA 12.1+ toolkit is installed")
            print("   - Ensure PyTorch was installed with CUDA support")
            return False
            
    except ImportError:
        print_status("PyTorch", False, "NOT INSTALLED")
        return False


def verify_monai_features():
    """Verify MONAI is installed with required features."""
    print_header("MONAI Framework Features")
    
    try:
        import monai
        from monai.utils import optional_import
        
        # MONAI version
        monai_version = monai.__version__
        print_status("MONAI Version", True, f"v{monai_version}")
        
        # Check optional dependencies
        features = {
            "nibabel": "NIfTI file support",
            "skimage": "Image transformations",
            "itk": "Advanced medical imaging",
            "tqdm": "Progress bars",
        }
        
        all_features = True
        for feature, description in features.items():
            module, available = optional_import(feature)
            print_status(f"  {feature:15s}", available, description)
            if not available:
                all_features = False
        
        return all_features
        
    except ImportError:
        print_status("MONAI", False, "NOT INSTALLED")
        return False


def verify_project_structure():
    """Verify project directory structure exists."""
    print_header("Project Directory Structure")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        "data_raw/dicom_files",
        "data_raw/annotations",
        "data_processed/nifti_images",
        "data_processed/nifti_labels_boxy",
        "data_processed/nifti_labels_medsam",
        "scripts",
        "configs",
        "models",
        "splits",
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(f"{dir_path:40s}", exists)
        if not exists:
            all_exist = False
    
    return all_exist


def verify_data_files():
    """Check if raw data files are present."""
    print_header("Raw Data Files")
    
    project_root = Path(__file__).parent
    
    csv_files = [
        "data_raw/annotations/TRAININGDATA.csv",
        "data_raw/annotations/COMPETITIONDATA.csv",
    ]
    
    data_present = True
    
    for csv_file in csv_files:
        full_path = project_root / csv_file
        exists = full_path.exists()
        
        if exists:
            # Count rows
            import pandas as pd
            try:
                df = pd.read_csv(full_path)
                row_count = len(df)
                print_status(f"{csv_file:50s}", True, f"{row_count:,} annotations")
            except Exception as e:
                print_status(f"{csv_file:50s}", True, "exists but couldn't read")
        else:
            print_status(f"{csv_file:50s}", False, "NOT FOUND")
            data_present = False
    
    # Check for DICOM files
    dicom_dir = project_root / "data_raw" / "dicom_files"
    if dicom_dir.exists():
        dicom_count = len(list(dicom_dir.rglob("*.dcm")))
        print_status(
            "DICOM files",
            dicom_count > 0,
            f"{dicom_count:,} files found"
        )
    else:
        print_status("DICOM files", False, "directory not found")
        data_present = False
    
    return data_present


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  ABDOMINAL EMERGENCY AI SEGMENTATION")
    print("  Environment Verification")
    print("=" * 70)
    
    results = {
        "Python Version": verify_python_version(),
        "Core Packages": verify_core_packages(),
        "PyTorch & CUDA": verify_pytorch_cuda(),
        "MONAI Features": verify_monai_features(),
        "Project Structure": verify_project_structure(),
        "Raw Data Files": verify_data_files(),
    }
    
    # Summary
    print_header("Verification Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        print_status(check, passed)
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✓ All checks passed! Environment is ready for development.")
        print("\nYou can now proceed with:")
        print("  Phase 1: DICOM to NIfTI conversion (scripts/dicom_to_nifti.py)")
        print("  Phase 1: Boxy label generation (scripts/make_boxy_labels.py)")
    else:
        print("✗ Some checks failed. Please review the output above.")
        print("\nTroubleshooting:")
        print("  - Reinstall missing packages with: pip install <package>")
        print("  - For CUDA issues, check NVIDIA drivers and PyTorch installation")
        print("  - Ensure raw data files are placed in data_raw/ directory")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
