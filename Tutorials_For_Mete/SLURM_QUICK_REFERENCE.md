# mesh-hpc Cluster Quick Reference

## Cluster Hardware Specs

| Component | Specification |
|-----------|--------------|
| **GPUs** | 2x NVIDIA RTX PRO 6000 Black (96GB VRAM each) |
| **CPUs** | Dual Intel Xeon 6530P (128 threads total) |
| **RAM** | 128 GB |
| **CUDA** | Max version 13.0 (Driver 580.95.05) |
| **Partition** | `mesh` |

## Essential SLURM Commands

### Job Submission
```bash
# Submit a job
sbatch script.sh

# Submit with custom job name
sbatch --job-name="my_experiment" script.sh

# Use the wrapper script (recommended)
./train.sh phase3_unet_training
./train.sh medsam_inference 0  # Force GPU 0
```

### Job Monitoring
```bash
# View your jobs
squeue -u $USER

# View all jobs in queue
squeue

# Check cluster resources
sinfo
sinfo -N -o "%n %c %m %G"

# Watch GPU usage
watch -n 1 nvidia-smi

# View job output in real-time
tail -f logs/train_12345.out

# Check completed jobs
sacct
```

### Job Control
```bash
# Cancel a job
scancel <JobID>

# Cancel all your jobs
scancel -u $USER

# Get detailed job info
scontrol show job <JobID>
```

## SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=mesh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --oversubscribe

# Your commands here
source ~/.bashrc
conda activate abdomen_scanner
python your_script.py
```

## Common SLURM Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `--job-name` | Job name | `--job-name=unet_training` |
| `--output` | Standard output file | `--output=logs/job_%j.out` |
| `--error` | Error output file | `--error=logs/job_%j.err` |
| `--time` | Max runtime (HH:MM:SS) | `--time=48:00:00` |
| `--partition` | Queue to use | `--partition=mesh` |
| `--gres` | Generic resources (GPU) | `--gres=gpu:1` |
| `--cpus-per-task` | CPU cores | `--cpus-per-task=16` |
| `--mem` | Memory | `--mem=90G` |
| `--nodes` | Number of nodes | `--nodes=1` |
| `--oversubscribe` | Allow oversubscription | `--oversubscribe` |

## Job States

| State | Code | Meaning |
|-------|------|---------|
| **PENDING** | PD | Waiting for resources |
| **RUNNING** | R | Currently executing |
| **COMPLETED** | CD | Finished successfully |
| **FAILED** | F | Exited with error |
| **CANCELLED** | CA | Manually cancelled |
| **TIMEOUT** | TO | Exceeded time limit |

## Best Practices

### Resource Requests
- **Don't request more than you need** - Helps queue efficiency
- **Memory:** Start with 64-90G, increase if needed
- **CPUs:** 8-16 cores for most tasks
- **Time:** Be realistic but add buffer (e.g., 48h for 24h job)

### GPU Selection
- Use wrapper script for **automatic GPU selection** (least loaded)
- Or specify GPU manually: `./train.sh my_exp 0` (force GPU 0)
- Check GPU usage: `watch -n 1 nvidia-smi`

### Monitoring
- Check logs regularly: `tail -f logs/*.out`
- Monitor queue: `squeue -u $USER`
- Cancel failed jobs promptly: `scancel <JobID>`

### Priority
- **AI security project** gets priority
- Plan long runs accordingly
- Communicate with team if urgent

## Common Issues & Solutions

### Issue: Job stays PENDING forever
```bash
# Check what's blocking it
squeue -u $USER
sinfo  # Check available resources

# Solution: Someone might be using GPUs
# Wait or request smaller resources
```

### Issue: Job FAILED immediately
```bash
# Check error log
cat logs/my_job_12345.err

# Common causes:
# - Python error (fix code)
# - Wrong conda environment
# - Missing files/data
```

### Issue: Out of memory
```bash
# Increase memory in SLURM script
#SBATCH --mem=120G  # (was 90G)

# Or reduce batch size in your code
```

### Issue: Job timeout
```bash
# Increase time limit
#SBATCH --time=96:00:00  # 4 days

# Or checkpoint your model more frequently
```

## Log File Patterns

SLURM automatically creates logs with job info:

```bash
logs/
├── train_12345.out     # Job 12345 output
├── train_12345.err     # Job 12345 errors
├── medsam_12346.out    # Job 12346 output
└── medsam_12346.err    # Job 12346 errors
```

Variables in filenames:
- `%x` - Job name
- `%j` - Job ID
- `%u` - Username

## Useful Aliases

Add to your `~/.bashrc` on cluster:

```bash
# SLURM shortcuts
alias sq='squeue -u $USER'
alias sqa='squeue'
alias si='sinfo'
alias gputop='watch -n 1 nvidia-smi'
alias logs='tail -f logs/*.out'

# Quick job cancel
scall() { scancel -u $USER; }
```

Then: `source ~/.bashrc`

## ⚠️ Critical Warning

**DO NOT** run scripts that repeatedly poll cluster status over SSH (e.g., in rapid loops):

```bash
# ❌ BAD - Don't do this!
while true; do
    squeue
    nvidia-smi
    sleep 1
done
```

**Why?** Excessive polling can overload the network and disrupt all users.

**✓ Instead:** Use `watch` command or check manually:
```bash
# ✓ Good
watch -n 5 nvidia-smi  # Updates every 5 seconds locally
```

## Resources

- **GitHub Repo:** https://github.com/JanosMozer/server-job-scheduler
- **SLURM Docs:** https://slurm.schedmd.com/
- **Team Notion:** https://growmesh.notion.site/slurm-job-scheduler

---

**Happy computing on mesh-hpc! 🚀**
