#!/bin/bash
#SBATCH --job-name=rsa_lss
#SBATCH --output=logs/rsa_lss_%j.out
#SBATCH --error=logs/rsa_lss_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G


# ── Paths — edit these ────────────────────────────────────────────────────────
SCRIPT_DIR=/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding
RESULTS_DIR=${SCRIPT_DIR}/rsa_results_masked
SUBJECT_TABLE=${SCRIPT_DIR}/subjects_paths.csv
HEATMAP_COORD="-44 36 28"   # DLPFC — change as needed

# ── Environment ───────────────────────────────────────────────────────────────
# Uncomment whichever applies to your setup:

# Option A — conda
module load anacondapy/2023.07-cuda
conda activate neu502b


# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "===================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Start time : $(date)"
echo "Script dir : ${SCRIPT_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "===================================="

python --version
python -c "import nilearn; print('nilearn', nilearn.__version__)"
python -c "import nibabel; print('nibabel', nibabel.__version__)"

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p ${RESULTS_DIR}
mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────────────
python ${SCRIPT_DIR}/rsa_lss.py \
    --subject-table   ${SUBJECT_TABLE} \
    --results-dir     ${RESULTS_DIR} \
    --n-jobs          ${SLURM_CPUS_PER_TASK} \
    --radius-mm       12.0 \
    --tr              2.0 \
    --stim-duration   1.0 \
    --heatmap-coord   ${HEATMAP_COORD}\
    --overwrite 

EXIT_CODE=$?

echo "===================================="
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "===================================="

exit ${EXIT_CODE}
