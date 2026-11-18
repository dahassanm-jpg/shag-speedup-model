#!/bin/bash
#SBATCH --job-name=shag-gromacs
#SBATCH --nodes=1                      # resubmit for each node count
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4              # ranks x threads = cores per node (tune once, then keep fixed)
#SBATCH --time=01:00:00
#SBATCH --exclusive
# GROMACS strong scaling on benchMEM (82k atoms) / benchRIB (2M atoms) from
# https://www.mpinat.mpg.de/grubmueller/bench .  Run once on CPU-only nodes and once with
# GPUs (add --gpus-per-node and -nb gpu -pme gpu) to repeat the heterogeneity test of Section V-A.
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
GMX=${GMX:-gmx_mpi}; TPR=${TPR:-benchRIB.tpr}; REPS=${REPS:-5}; NB=${NB:-cpu}
N=$SLURM_JOB_NUM_NODES
for rep in $(seq 1 $REPS); do
  D=results/gromacs/$(basename $TPR .tpr)-$NB/nodes$N/rep$rep; mkdir -p $D
  [ $rep = 1 ] && source common/metadata.sh $D
  srun $GMX mdrun -s $TPR -deffnm $D/md -nsteps 20000 -resethway -noconfout -nb $NB \
       -ntomp $SLURM_CPUS_PER_TASK -pin on -dlb yes
done
