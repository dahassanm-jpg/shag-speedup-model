#!/bin/bash
#SBATCH --job-name=shag-micro
#SBATCH --nodes=1                 # resubmit with --nodes=N for every target node count
#SBATCH --ntasks-per-node=1       # set to cores (CPU codes) or GPUs (DL) per node
#SBATCH --time=00:30:00
#SBATCH --exclusive
# Communication micro-benchmarks + noise benchmark (Section IV-E steps 4-5).
# Requires the OSU Micro-Benchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/) built with the
# SAME MPI as the applications; set OSU_DIR. Message sizes should bracket the application's messages.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OSU_DIR=${OSU_DIR:?set OSU_DIR to .../libexec/osu-micro-benchmarks/mpi}
NP=$SLURM_NTASKS; OUT=results/microbench/np${NP}; mkdir -p $OUT
source common/metadata.sh $OUT
for rep in 1 2 3; do
  srun $OSU_DIR/collective/osu_allreduce -m 8:268435456 -i 200 > $OUT/osu_allreduce_rep$rep.txt
  srun $OSU_DIR/collective/osu_alltoall  -m 8:4194304   -i 200 > $OUT/osu_alltoall_rep$rep.txt
  srun $OSU_DIR/collective/osu_barrier   -i 2000             > $OUT/osu_barrier_rep$rep.txt
  srun ./microbench/fwq 2000 200000 $OUT/fwq_samples_rep$rep.csv > $OUT/fwq_rep$rep.txt
done
if [ "$NP" -eq 2 ]; then   # point-to-point latency/bandwidth across two nodes
  srun $OSU_DIR/pt2pt/osu_latency > $OUT/osu_latency.txt
  srun $OSU_DIR/pt2pt/osu_bw      > $OUT/osu_bw.txt
fi
