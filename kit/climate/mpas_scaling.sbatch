#!/bin/bash
#SBATCH --job-name=shag-mpas
#SBATCH --nodes=1                      # resubmit for each node count
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --exclusive
# MPAS-Atmosphere strong scaling on the 120 km quasi-uniform mesh (x1.40962), the configuration of
# Heinzeller et al. (2016) Table D1.  Needs a mesh partition file x1.40962.graph.info.part.<NTASKS>
# (gpmetis) for every task count, an initial-conditions file, and namelist.atmosphere with
# config_run_duration='1_00:00:00'.  The log file contains MPAS's built-in timer table.
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
MPAS=${MPAS:-./atmosphere_model}; RUNDIR=${RUNDIR:-mpas_120km}; REPS=${REPS:-3}
NT=$SLURM_NTASKS
for rep in $(seq 1 $REPS); do
  D=results/mpas/120km/np$NT/rep$rep; mkdir -p $D
  [ $rep = 1 ] && source common/metadata.sh $D
  ( cd $RUNDIR && srun $MPAS && cp log.atmosphere.0000.out ../$D/ )
done
