#!/bin/bash --login
#SBATCH --job-name=neat_mnist
#SBATCH --partition=general-long-gpu
#SBATCH --gpus=h200:1
#SBATCH --gpu-bind=single:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=03:59:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Load Python
module load Python/3.11.3

# Activate your virtual environment
source ~/Documents/tensorneat/venv/bin/activate

# Navigate into the tensorneat library
cd ~/Documents/tensorneat/

# Run MNIST training
python run_mnist.py

