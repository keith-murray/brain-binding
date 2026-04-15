#!/bin/bash
#SBATCH --job-name=rsa_analysis
#SBATCH --output=logs/rsa_analysis_%j.out
#SBATCH --error=logs/rsa_analysis_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR=/usr/people/yl0124/projects/NEU502B/neu502b-2025/binding
RESULTS_DIR=${SCRIPT_DIR}/rsa_results_masked

# SUBJECTS="sub-001 sub-002"
SUBJECTS="sub-001 sub-002 sub-008"

# Searchlight options
RADIUS_MM=12.0
DLPFC_COORD="-44 36 28"

# Set to "--use-ranks" to rank-transform y in OLS (Spearman-equivalent)
# Leave empty "" for raw correlation distances (default, more interpretable)
USE_RANKS=""
# USE_RANKS="--use-ranks"

# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

module load anacondapy/2023.07-cuda
conda activate neu502b

# ══════════════════════════════════════════════════════════════════════════════
# Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

echo "===================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Mem        : 64G"
echo "Start time : $(date)"
echo "Script dir : ${SCRIPT_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Subjects   : ${SUBJECTS}"
echo "Radius     : ${RADIUS_MM} mm"
echo "DLPFC coord: ${DLPFC_COORD}"
echo "Use ranks  : ${USE_RANKS:-no}"
echo "===================================="

python --version
python -c "import nilearn; print('nilearn', nilearn.__version__)"
python -c "import nibabel; print('nibabel', nibabel.__version__)"

mkdir -p ${RESULTS_DIR}
mkdir -p logs

# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

python ${SCRIPT_DIR}/rsa_analysis.py \
    --results-dir  ${RESULTS_DIR} \
    --subjects     ${SUBJECTS} \
    --radius-mm    ${RADIUS_MM} \
    --dlpfc-coord  ${DLPFC_COORD} \
    --n-jobs       ${SLURM_CPUS_PER_TASK} \
    ${USE_RANKS}

EXIT_CODE=$?

echo "===================================="
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "===================================="

exit ${EXIT_CODE}