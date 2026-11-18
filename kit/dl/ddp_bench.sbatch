#!/bin/bash
#SBATCH --job-name=shag-ddp
#SBATCH --nodes=1                      # resubmit for 1,2,4,8,... nodes
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --exclusive
# ResNet-50 (weak: 32 images/GPU) and BERT-large-like encoder (strong: global batch 512, seq 128).
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
G=${SLURM_GPUS_PER_NODE:-4}; N=$SLURM_JOB_NUM_NODES; REPS=${REPS:-3}
HEAD=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
D=results/dl/nodes$N; mkdir -p $D; source common/metadata.sh $D
for rep in $(seq 1 $REPS); do
  srun torchrun --nnodes=$N --nproc-per-node=$G --rdzv-backend=c10d --rdzv-endpoint=$HEAD:29500 \
      dl/ddp_bench.py --model resnet50 --mode weak --per-rank-batch 32 --amp --steps 50 --out $D/resnet50_weak.jsonl
  srun torchrun --nnodes=$N --nproc-per-node=$G --rdzv-backend=c10d --rdzv-endpoint=$HEAD:29500 \
      dl/ddp_bench.py --model bert-large --mode strong --global-batch 512 --seq-len 128 --amp --steps 30 \
      --out $D/bert_strong.jsonl
done
# Single-GPU baseline (P = 1): run once with --nodes=1 and G=1 (set SLURM_GPUS_PER_NODE=1).
