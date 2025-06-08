#!/bin/bash
#SBATCH --job-name=test_enhanced_features
#SBATCH --partition=gpu
#SBATCH --nodelist=rack7n06
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --output=test_enhanced_features_%j.out
#SBATCH --error=test_enhanced_features_%j.err

# Print job info
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Check Python environment
echo "=== Python Environment ==="
which python
python --version
echo ""

# Check PyTorch/CUDA
echo "=== PyTorch/CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

# Set up environment
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run the test script with debugpy
echo "=== Running Enhanced Features Test ==="
echo "Starting debugpy server on 0.0.0.0:5678..."
echo "Connect your debugger to rack7n06:5678"
echo ""

# Run with debugpy (will wait for debugger connection)
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client test_enhanced_features.py

echo ""
echo "=== Test Completed at $(date) ===" 