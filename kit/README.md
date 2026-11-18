# S-HAG cluster experiment kit

Runs the controlled measurement protocol of Section V-D on a Slurm cluster, then produces
component-calibrated S-HAG predictions next to Amdahl, Gustafson and USL.

## What it measures

| Application | Script | Scaling | Component timers used by S-HAG |
|---|---|---|---|
| LAMMPS (LJ, 32k atoms/rank weak, 256k atoms strong) | `md/lammps_scaling.sbatch` | strong + weak | Pair/Neigh/Modify (parallel), Comm, Kspace, Output+Other (serial), Pair max−avg (imbalance) |
| GROMACS benchMEM / benchRIB | `md/gromacs_scaling.sbatch` | strong, CPU-only and GPU nodes | cycle-accounting table: Comm./Wait/PME-redist (comm), Update/Constraints/Write (serial) |
| ResNet-50, BERT-large-like encoder | `dl/ddp_bench.sbatch` → `dl/ddp_bench.py` | weak (ResNet), strong (BERT) | compute (no_sync), optimizer (serial), gradient all-reduce, max−mean over ranks |
| MPAS-A 120 km | `climate/mpas_scaling.sbatch` | strong | built-in timer table (halo exchange, I/O, integration) |
| CESM B1850 | `climate/CESM_PE_SWEEP.md` | strong, PE-layout sweep | CIME timing file: component run times, CPL COMM time |
| Network + noise | `microbench/microbench.sbatch` (OSU + `fwq.c`) | all target P | all-reduce / all-to-all / barrier latency, FWQ noise σₐ |

## Run order

1. Build `microbench/fwq` (`mpicc -O1 -o microbench/fwq microbench/fwq.c -lm`) and the OSU Micro-Benchmarks with
   the same MPI as the applications.
2. For every node count N in {1, 2, 4, 8, 16, …}: `sbatch --nodes=N <script>` for each application and for
   `microbench.sbatch`. Use exclusive nodes, the same software stack and the same pinning for all runs.
   At least 5 repetitions for CPU codes, 3 for DL and climate.
3. `python collect.py results --allreduce-bytes <gradient bytes> --alltoall-bytes <FFT block bytes>`
   → `components.csv`, `microbench.csv`.
4. `python analyze_own_runs.py components.csv [microbench_mapped.csv]`
   → `results/own_predictions.csv`, `results/own_accuracy.csv`.

The P = 1 run of every series is the reference resource of Section IV (one core, one GPU, or one node,
stated explicitly). Calibration uses the smallest ⌈n/2⌉ resource counts; the larger ones are held out.

## What was tested here (2 CPU cores, no GPU, no scheduler)

| Component | Test |
|---|---|
| `microbench/fwq.c` | compiled with MPICH and run on 2 ranks |
| `dl/ddp_bench.py` | run with torchrun on CPU/gloo, 1 and 2 processes, ResNet-18 and a small Transformer |
| `parsers.parse_lammps` | on a real LAMMPS log (LJ, 32,000 atoms) |
| `parsers.parse_ddp`, `parse_fwq` | on the real outputs above |
| `analyze_own_runs.py` | on the real FABRIC LAMMPS timers (216 runs); reproduces the component-calibrated results of the main pipeline |
| `collect.py` | on a results tree containing the real outputs above |
| `*.sbatch` | `bash -n` syntax check only |
| `parse_gromacs`, `parse_mpas`, `parse_cesm`, `parse_osu` | **not run on real output**: check the first parsed file by hand |

Nothing in this kit has been run on a multi-node cluster or a GPU.
