#!/bin/bash
# Record hardware/software metadata for every job (Section V-D, reporting rule R1).
# Usage: source common/metadata.sh <out_dir>
OUT=${1:-.}; mkdir -p "$OUT"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job: ${SLURM_JOB_ID:-none}"
  echo "nodes: ${SLURM_JOB_NUM_NODES:-1}"; echo "nodelist: ${SLURM_JOB_NODELIST:-$(hostname)}"
  echo "--- lscpu"; lscpu 2>/dev/null
  echo "--- memory"; free -g 2>/dev/null
  echo "--- gpus"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo none
  echo "--- interconnect"; ibstat 2>/dev/null | grep -E "CA type|Rate|Link layer" || echo "ibstat unavailable"
  echo "--- mpi"; mpirun --version 2>/dev/null | head -2
  echo "--- modules"; module list 2>&1 | head -40
  echo "--- kernel"; uname -a
} > "$OUT/metadata.txt"
