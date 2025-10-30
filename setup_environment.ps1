# ============================================================================
# Abdominal Emergency AI Segmentation - Environment Setup Script
# ============================================================================
# This script sets up the conda environment for the project with all required
# dependencies for Phase 1, 2, and 3 of the development roadmap.
#
# Prerequisites:
# - Anaconda3 installed and registered as default Python
# - NVIDIA A6000 GPUs with CUDA 12.1+ drivers
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Abdomen Scanner Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project root
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Step 0.1: Creating dedicated conda environment..." -ForegroundColor Yellow
Write-Host "This isolates project dependencies and prevents conflicts." -ForegroundColor Gray
Write-Host ""

# Check if environment already exists
$envExists = conda env list | Select-String "abdomen_scanner"

if ($envExists) {
    Write-Host "Environment 'abdomen_scanner' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and recreate it? (y/n)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        conda env remove -n abdomen_scanner -y
    } else {
        Write-Host "Using existing environment." -ForegroundColor Green
        conda activate abdomen_scanner
        Write-Host ""
        Write-Host "Environment activated. Run 'python verify_setup.py' to check installation." -ForegroundColor Green
        exit 0
    }
}

Write-Host "Creating environment from environment.yml..." -ForegroundColor Yellow
Write-Host "This includes:" -ForegroundColor Gray
Write-Host "  - Python 3.10" -ForegroundColor Gray
Write-Host "  - PyTorch 2.0+ with CUDA 12.1 support (for A6000 GPUs)" -ForegroundColor Gray
Write-Host "  - Medical imaging libraries (pydicom, SimpleITK, nibabel)" -ForegroundColor Gray
Write-Host "  - MONAI framework for medical AI" -ForegroundColor Gray
Write-Host "  - Data science stack (pandas, numpy, scikit-learn)" -ForegroundColor Gray
Write-Host ""

# Create environment
conda env create -f environment.yml

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Environment created successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Activate the environment:" -ForegroundColor White
    Write-Host "   conda activate abdomen_scanner" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Verify the installation:" -ForegroundColor White
    Write-Host "   python verify_setup.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "3. Start Phase 1 development:" -ForegroundColor White
    Write-Host "   - DICOM to NIfTI conversion" -ForegroundColor Gray
    Write-Host "   - Boxy label generation" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Environment creation failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Ensure you're running this in Anaconda PowerShell Prompt" -ForegroundColor White
    Write-Host "2. Check your internet connection (packages need to download)" -ForegroundColor White
    Write-Host "3. Try manual installation (see SETUP_MANUAL.md)" -ForegroundColor White
    Write-Host ""
    exit 1
}
