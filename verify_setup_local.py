"""
Local Environment Verification Script (No GPU Required)
========================================================

This script verifies your LOCAL Windows setup for development.
It does NOT check for GPUs since you'll use the remote cluster for that.

Run this on your Windows laptop after setting up the local conda environment.

Usage:
    conda activate abdomen_scanner
    python verify_setup_local.py
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
        
        # Phase 2: Image Processing
        "cv2": "OpenCV (image operations)",
        "PIL": "Pillow (image processing)",
        "skimage": "scikit-image (transformations)",
        "sklearn": "scikit-learn (data splitting)",
        
        # Phase 3: Deep Learning (will use cluster GPUs)
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


def verify_pytorch_local():
    """Verify PyTorch is installed (GPU check skipped for local)."""
    print_header("PyTorch Configuration (Local)")
    
    try:
        import torch
        
        pytorch_version = torch.__version__
        print_status("PyTorch Version", True, f"v{pytorch_version}")
        
        print("\n⚠️  Note: GPU checks skipped - you'll use the mesh-hpc cluster for GPU work.")
        print("   This local environment is for development and data preparation only.")
        
        return True
            
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
        "Tutorials_For_Mete",
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(f"{dir_path:40s}", exists)
        if not exists:
            all_exist = False
    
    return all_exist


def verify_slurm_scripts():
    """Check if SLURM job scripts exist."""
    print_header("SLURM Job Scripts")
    
    project_root = Path(__file__).parent
    
    slurm_scripts = [
        "slurm_test_gpu.sh",
        "slurm_dicom_to_nifti.sh",
        "slurm_make_boxy_labels.sh",
        "slurm_medsam_infer.sh",
        "slurm_train_unet.sh",
    ]
    
    all_exist = True
    
    for script in slurm_scripts:
        full_path = project_root / script
        exists = full_path.exists()
        print_status(f"{script:35s}", exists)
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  ABDOMINAL EMERGENCY AI SEGMENTATION")
    print("  Local Environment Verification (Windows)")
    print("=" * 70)
    
    results = {
        "Python Version": verify_python_version(),
        "Core Packages": verify_core_packages(),
        "PyTorch (Local)": verify_pytorch_local(),
        "MONAI Features": verify_monai_features(),
        "Project Structure": verify_project_structure(),
        "SLURM Scripts": verify_slurm_scripts(),
    }
    
    # Summary
    print_header("Verification Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        print_status(check, passed)
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✓ Local environment is ready for development!")
        print("\nYou can now:")
        print("  1. Develop and test code locally")
        print("  2. Prepare data (Phase 1 scripts)")
        print("  3. Upload to mesh-hpc cluster for GPU work")
        print("\nNext steps:")
        print("  - Read: Tutorials_For_Mete/HPC_CLUSTER_SETUP.md")
        print("  - Get SSH credentials from admin")
        print("  - Install Tailscale VPN")
        print("  - Connect to mesh-hpc and set up cluster environment")
    else:
        print("✗ Some checks failed. Please review the output above.")
        print("\nTroubleshooting:")
        print("  - Reinstall missing packages with: pip install <package>")
        print("  - Ensure you activated the conda environment")
        print("  - Run setup_environment.ps1 if not done yet")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
