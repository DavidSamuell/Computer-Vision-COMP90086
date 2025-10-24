#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load necessary modules
module load Python/3.8.6
module load CUDA/11.6.0

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/src

# Change to the project directory
cd /data/projects/punim0478/setiawand/Computer-Vision-COMP90086

# Create output directory
mkdir -p outputs/nutrition5k_paper_experiments

# Run the experiment script - the exact configuration from the Nutrition5k paper
python src/nutrition5k_paper_experiment.py
