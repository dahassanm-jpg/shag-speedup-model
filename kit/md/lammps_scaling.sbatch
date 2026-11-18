#!/bin/bash
#SBATCH --job-name=shag-lammps
#SBATCH --nodes=1                      # resubmit for each node count: sbatch --nodes=N ...
#SBATCH --ntasks-per-node=32           # = physical cores per node
#SBATCH --time=02:00:00
#SBATCH --exclusive
# LAMMPS strong + weak scaling with per-section timers (Section V-D run matrix).
# Strong: fixed 256,000-atom LJ (x=y=z=2). Weak: 32,000 atoms per MPI rank.
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
LMP=${LMP:-lmp}; REPS=${REPS:-5}; STEPS=${STEPS:-2000}
NMAX=$SLURM_NTASKS
for P in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
  [ $P -gt $NMAX ] && break
  for rep in $(seq 1 $REPS); do
    for mode in strong weak; do
      if [ $mode = strong ]; then X=2; Y=2; Z=2
      else  # replicate the 32k-atom cell P times, as close to cubic as possible
        read X Y Z < <(python3 -c "
import itertools,math;P=$P
b=min(((x,y,P//(x*y)) for x in range(1,P+1) for y in range(1,P+1) if P%(x*y)==0),key=lambda t:max(t)-min(t))
print(*b)")
      fi
      D=results/lammps/$mode/lj/np$P/rep$rep; mkdir -p $D
      [ $rep = 1 ] && source common/metadata.sh $D
      srun --ntasks=$P --cpu-bind=cores $LMP -in md/inputs/in.lj -var x $X -var y $Y -var z $Z \
           -var nsteps $STEPS -log $D/log.lammps -screen none
    done
  done
done
