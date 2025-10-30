# 🖥️ HPC CLUSTER SETUP GUIDE - Mesh HPC with SLURM

## 🎯 NEW: Remote GPU Cluster Access

**Important Change:** You're NOT using local GPUs. Instead, you'll connect to a **remote HPC cluster** (mesh-hpc) that has GPUs available for 6 months.

---

## 📚 What You Need to Understand First

### What is HPC (High-Performance Computing)?
A **remote server cluster** with powerful GPUs that multiple team members can share. Think of it like a super-powerful computer you access over the internet.

### What is SLURM?
A **job scheduler** that manages who gets to use which GPU and when. Instead of running code directly, you:
1. Write a script describing what you want to run
2. Submit it to SLURM (`sbatch my_script.sh`)
3. SLURM queues your job and runs it when a GPU is free
4. You get the results when it's done

**Why?** Multiple people can't use the same GPU at once. SLURM prevents conflicts and ensures fair access.

### What is Tailscale?
A **VPN tool** that lets you securely connect to the mesh-hpc cluster from your Windows laptop.

---

## 🛠️ Part 1: Local Setup (Your Windows Laptop)

You still need a Python environment on your laptop for:
- **Writing code** and scripts
- **Testing small things** before sending to cluster
- **Data preprocessing** (DICOM → NIfTI conversion)
- **Visualizing results** from the cluster

### Step 1: Install Tailscale (VPN Access)

1. **DM the team lead** to get:
   - Tailscale invitation
   - Cluster hostname/IP address
   - Your SSH credentials (username/password)

2. **Install Tailscale:**
   - Download from: https://tailscale.com/download/windows
   - Install and sign in with the invitation link

3. **Connect to mesh-hpc network:**
   - Tailscale will show a green checkmark when connected

### Step 2: Set Up Local Python Environment

Even though you'll train on the cluster, you need Python locally for development.

**Open Anaconda PowerShell Prompt:**

```powershell
# Navigate to project
cd "c:\Users\User\Desktop\Puzzles Software\Projects\OmniScan Medical\Abdomen-Scanner\project_root"

# Create local environment (for development only)
conda env create -f environment.yml

# Activate it
conda activate abdomen_scanner

# Verify local setup
python verify_setup_local.py
```

**Note:** This local environment won't have GPU access - that's fine! You'll use the cluster for GPU work.

---

## 🖥️ Part 2: Cluster Setup (Remote mesh-hpc)

### Step 3: Connect to the Cluster via SSH

**Open PowerShell** (regular PowerShell, not Anaconda):

```powershell
# Connect to cluster (replace with your actual credentials)
ssh your-username@mesh-hpc-hostname

# Enter password when prompted
```

**You're now inside the cluster!** The terminal prompt will change to show the cluster name.

### Step 4: Set Up Your Cluster Environment

Once connected to the cluster:

```bash
# Navigate to your home directory
cd ~

# Create project directory
mkdir -p abdomen-scanner
cd abdomen-scanner

# Load conda module (if available)
module load anaconda3  # or miniconda3

# Create environment on cluster
conda env create -f environment.yml

# Activate it
conda activate abdomen_scanner

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## 📤 Part 3: Transferring Files to Cluster

You'll develop code on your Windows laptop, then send it to the cluster for training.

### Method 1: SCP (Secure Copy)

**From your Windows PowerShell:**

```powershell
# Upload a single file
scp "path\to\local\file.py" your-username@mesh-hpc:~/abdomen-scanner/

# Upload entire directory
scp -r "path\to\local\scripts" your-username@mesh-hpc:~/abdomen-scanner/

# Download results from cluster
scp your-username@mesh-hpc:~/abdomen-scanner/results/model.pth "C:\Users\User\Desktop\"
```

### Method 2: VS Code Remote SSH (Recommended)

**Much easier!** Edit files directly on the cluster through VS Code.

1. **Install VS Code extension:** "Remote - SSH"
2. **Press:** `Ctrl+Shift+P`
3. **Type:** "Remote-SSH: Connect to Host"
4. **Enter:** `your-username@mesh-hpc-hostname`
5. **Enter password**
6. **Open folder:** `/home/your-username/abdomen-scanner`

Now you can edit files on the cluster as if they were local!

---

## 🚀 Part 4: Using SLURM Job Scheduler

### Understanding SLURM Commands

| Command | What It Does |
|---------|-------------|
| `sbatch script.sh` | Submit a job to the queue |
| `squeue` | See all running/pending jobs |
| `squeue -u $USER` | See only YOUR jobs |
| `scancel <job-id>` | Cancel a job |
| `sinfo` | Show available GPUs/nodes |
| `sacct` | See completed job history |

### Step 5: Create SLURM Job Scripts

I'll create example SLURM scripts for each phase of your project.

**Read the SLURM guide:** https://growmesh.notion.site/slurm-job-scheduler

---

## 📝 Example SLURM Scripts

### Script 1: Phase 1 - DICOM to NIfTI Conversion

This might not need a GPU, so it can run on CPU nodes:

```bash
#!/bin/bash
#SBATCH --job-name=dicom_to_nifti
#SBATCH --output=logs/dicom_%j.out
#SBATCH --error=logs/dicom_%j.err
#SBATCH --time=04:00:00          # 4 hours max
#SBATCH --cpus-per-task=8        # 8 CPU cores
#SBATCH --mem=32G                # 32GB RAM

# Load conda
source ~/.bashrc
conda activate abdomen_scanner

# Run Phase 1 script
echo "Starting DICOM to NIfTI conversion..."
python scripts/dicom_to_nifti.py

echo "Done!"
```

**Submit with:** `sbatch slurm_dicom_to_nifti.sh`

### Script 2: Phase 2 - MedSAM Inference (Needs GPU)

```bash
#!/bin/bash
#SBATCH --job-name=medsam_infer
#SBATCH --output=logs/medsam_%j.out
#SBATCH --error=logs/medsam_%j.err
#SBATCH --time=12:00:00          # 12 hours
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load conda
source ~/.bashrc
conda activate abdomen_scanner

# Check GPU access
echo "Checking GPU availability..."
nvidia-smi

# Run MedSAM inference
echo "Starting MedSAM pseudo-mask generation..."
python scripts/medsam_infer.py --batch-size 4

echo "MedSAM inference complete!"
```

**Submit with:** `sbatch slurm_medsam_infer.sh`

### Script 3: Phase 3 - 3D U-Net Training (Primary Use Case)

```bash
#!/bin/bash
#SBATCH --job-name=unet_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00          # 2 days max
#SBATCH --gpus=1                 # Request 1 GPU (multi-GPU not working yet)
#SBATCH --cpus-per-task=16       # More CPUs for data loading
#SBATCH --mem=128G               # Large RAM for 3D data

# Load conda
source ~/.bashrc
conda activate abdomen_scanner

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Check GPU
echo "GPU Information:"
nvidia-smi

# Run training
echo "Starting 3D U-Net training..."
python scripts/train_monai.py \
    --config configs/config.yaml \
    --epochs 300 \
    --batch-size 2 \
    --num-workers 8

echo "Training complete! Model saved to models/"
```

**Submit with:** `sbatch slurm_train_unet.sh`

---

## 🔍 Monitoring Your Jobs

### Check Job Status

```bash
# See all your jobs
squeue -u $USER

# Output:
# JOBID   PARTITION   NAME          USER    STATE    TIME  NODES
# 12345   gpu         unet_training yourname RUNNING  1:23:45  1
```

**Job States:**
- `PENDING (PD)` - Waiting for GPU to be free
- `RUNNING (R)` - Currently executing
- `COMPLETED (CD)` - Finished successfully
- `FAILED (F)` - Crashed/errored

### View Job Output in Real-Time

```bash
# Watch the output log (updates live)
tail -f logs/train_12345.out

# Press Ctrl+C to stop watching
```

### Cancel a Job

```bash
# Cancel by job ID
scancel 12345

# Cancel all your jobs
scancel -u $USER
```

---

## ⚠️ Important Cluster Rules

### Priority System

Your **AI security project** gets priority. If someone from that project needs the GPU:
- Your job might be paused or queued longer
- Plan long training runs accordingly
- Communicate with the team

### Resource Etiquette

1. **Don't hog resources:**
   - Request only what you need (`--gpus=1` unless you really need more)
   - Set realistic time limits (`--time`)
   
2. **Test locally first:**
   - Debug code on your laptop
   - Only submit working code to cluster
   
3. **Monitor your jobs:**
   - Check logs regularly: `tail -f logs/train_*.out`
   - Cancel failed jobs: `scancel <job-id>`
   
4. **Ask before installing:**
   - Only install simple packages (`pip install`)
   - For system-level changes (CUDA, drivers), DM the admin first

---

## 📊 Typical Workflow

Here's how you'll work day-to-day:

### On Your Windows Laptop:

1. **Write/edit Python code** in VS Code
2. **Test small things** in your local conda environment
3. **Prepare data** (Phase 1: DICOM → NIfTI)

### On the Cluster:

4. **Upload code** via SCP or VS Code Remote SSH
5. **Write SLURM script** for the task
6. **Submit job:** `sbatch slurm_train.sh`
7. **Monitor:** `squeue -u $USER`
8. **Check logs:** `tail -f logs/train_*.out`
9. **Download results** when done

### Example Day:

```powershell
# Morning: Develop code locally
cd "C:\...\project_root"
conda activate abdomen_scanner
# Edit scripts/train_monai.py in VS Code

# Upload to cluster
scp scripts/train_monai.py user@mesh-hpc:~/abdomen-scanner/scripts/

# SSH into cluster
ssh user@mesh-hpc

# On cluster: Submit training job
cd ~/abdomen-scanner
sbatch slurm_train_unet.sh

# Check status
squeue -u $USER

# Logout (job keeps running!)
exit

# Later: Check if done
ssh user@mesh-hpc
squeue -u $USER  # If not listed, it's done!

# Download trained model
exit
scp user@mesh-hpc:~/abdomen-scanner/models/best_model.pth "C:\Users\User\Desktop\"
```

---

## 🛠️ Setup Checklist

### Local Setup (Windows):
- [ ] Tailscale installed and connected
- [ ] SSH credentials received from admin
- [ ] Anaconda installed locally
- [ ] Local conda environment created (`abdomen_scanner`)
- [ ] VS Code with Remote-SSH extension

### Cluster Setup (mesh-hpc):
- [ ] Successfully SSH'd into cluster
- [ ] Project directory created (`~/abdomen-scanner`)
- [ ] Conda environment created on cluster
- [ ] GPU access verified (`nvidia-smi` works)
- [ ] SLURM guide read (https://growmesh.notion.site/slurm-job-scheduler)

### First Job Test:
- [ ] Created a simple test SLURM script
- [ ] Submitted with `sbatch`
- [ ] Monitored with `squeue`
- [ ] Viewed output with `tail -f`
- [ ] Job completed successfully

---

## 📚 Resources

- **SLURM Official Docs:** https://slurm.schedmd.com/
- **Team SLURM Guide:** https://growmesh.notion.site/slurm-job-scheduler
- **Tailscale Setup:** https://tailscale.com/kb/
- **VS Code Remote SSH:** https://code.visualstudio.com/docs/remote/ssh

---

## 🆘 Troubleshooting

### Problem: Can't SSH to cluster
**Solution:** 
1. Check Tailscale is connected (green icon)
2. Verify hostname/IP with admin
3. Try: `ping mesh-hpc-hostname`

### Problem: "Permission denied" on cluster
**Solution:** Check you're using correct username/password from admin

### Problem: Job stays in PENDING forever
**Solution:**
1. Check available GPUs: `sinfo`
2. Someone else might be using them
3. Your job might be lower priority (AI security project first)

### Problem: Job FAILED immediately
**Solution:**
1. Check error log: `cat logs/train_<job-id>.err`
2. Usually a Python error - fix code and resubmit

---

**Ready to use the cluster! 🚀**

**Next:** Read the SLURM guide, get your SSH credentials, and I'll help you submit your first job!
