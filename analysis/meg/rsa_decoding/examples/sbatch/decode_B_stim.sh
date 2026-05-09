#!/bin/bash
#SBATCH --job-name=decode_B_stim
#SBATCH --output=logs/decode_B_stim_%j.out
#SBATCH --error=logs/decode_B_stim_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# ── Paths — edit these ────────────────────────────────────────────────────────
SCRIPT_DIR=/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding/examples

# ── Environment ───────────────────────────────────────────────────────────────
module load anacondapy/2023.07-cuda
conda activate neu502b

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p ${SCRIPT_DIR}/logs

python ${SCRIPT_DIR}/decode_meg.py --experiment B_stim
