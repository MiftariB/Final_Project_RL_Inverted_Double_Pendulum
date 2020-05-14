#!/bin/bash
#
#SBATCH --job-name=RL
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --mem-per-cpu=32g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

python3 InvertedPendulum.py
