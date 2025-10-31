# 📚 Documentation Index

> **Organized guides for the Abdomen Scanner project**

---

## 🚀 Getting Started

### 1. [QUICKSTART.md](QUICKSTART.md)
**⚡ Start here!** - Get training in 3 commands
- Prerequisites checklist
- Three phases in three commands
- Common issues & solutions

### 2. [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)
**📋 Complete workflow** - Deep dive into each phase
- Detailed Phase 1-3 breakdown
- Technical specifications
- Expected timelines
- Success metrics

---

## 🛠️ Setup & Configuration

### 3. [SETUP_GUIDE.md](SETUP_GUIDE.md)
**Environment setup** - How to install and configure
- Prerequisites (hardware/software)
- Conda environment setup
- Verification steps
- Project structure

### 4. [../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md](../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md)
**HPC cluster access** - Connect to mesh-hpc
- Tailscale VPN setup
- SSH configuration
- SLURM basics
- File transfers

### 5. [../Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md](../Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md)
**SLURM commands** - Cluster hardware specs and common commands
- Job submission/monitoring
- Resource allocation
- Troubleshooting

---

## 💻 Reference

### 6. [DATA_FORMAT.md](DATA_FORMAT.md) ⭐ **NEW**
**Dataset structure** - Critical information about annotation format
- Excel file structure (TRAIININGDATA sheet - 3 i's!)
- 11→6 class mapping with sample counts
- Bounding box + boundary slice format
- Class imbalance details (Class 5/6 have only 54 annotations each)
- Z-axis validation logic

### 7. [COMMANDS.md](COMMANDS.md)
**Command cheat sheet** - Quick reference for all common operations
- Running phases
- Monitoring jobs
- File operations
- Git & DVC commands
- Emergency commands

### 8. [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)
**Recent changes** - What was updated in October 2025
- Complete list of implemented scripts
- Technical improvements
- File organization

---

## 📖 Reading Order

**If this is your first time:**
1. Read [QUICKSTART.md](QUICKSTART.md) - Get the big picture
2. Read [DATA_FORMAT.md](DATA_FORMAT.md) - Understand the dataset ⭐
3. Skim [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Understand the phases
4. Follow [HPC_CLUSTER_SETUP.md](../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md) - Set up access
5. Keep [COMMANDS.md](COMMANDS.md) open - Reference as needed

**If you're ready to execute:**
1. [QUICKSTART.md](QUICKSTART.md) - Run the 3 commands
2. [COMMANDS.md](COMMANDS.md) - Monitor your jobs
3. [SLURM_QUICK_REFERENCE.md](../Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md) - SLURM help

**If you need deep technical details:**
1. [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Full workflow
2. [SETUP_GUIDE.md](SETUP_GUIDE.md) - Configuration details
3. [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) - Implementation notes

---

## 🔗 Quick Links

| What You Need | Go To |
|---------------|-------|
| "How do I start?" | [QUICKSTART.md](QUICKSTART.md) |
| "What are the phases?" | [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) |
| "How do I access the cluster?" | [../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md](../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md) |
| "What commands do I run?" | [COMMANDS.md](COMMANDS.md) |
| "How do I monitor jobs?" | [COMMANDS.md](COMMANDS.md#-monitoring) |
| "What changed recently?" | [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) |

---

## 📂 File Locations

All documentation is organized as follows:

```
abdomen-scanner/
├── README.md                         # Main project overview
├── docs/                             # THIS DIRECTORY
│   ├── README.md                     # You are here (index)
│   ├── QUICKSTART.md                 # Quick start guide
│   ├── PROJECT_ROADMAP.md            # Complete workflow
│   ├── SETUP_GUIDE.md                # Environment setup
│   ├── COMMANDS.md                   # Command reference
│   └── UPDATE_SUMMARY.md             # Change log
└── Tutorials_For_Mete/               # Beginner tutorials
    ├── BEGINNERS_SETUP.md            # First-time setup
    ├── HPC_CLUSTER_SETUP.md          # Cluster access
    └── SLURM_QUICK_REFERENCE.md      # SLURM commands
```

---

**Need help?** Start with [QUICKSTART.md](QUICKSTART.md) or ask the project lead.

**Last Updated:** October 31, 2025
