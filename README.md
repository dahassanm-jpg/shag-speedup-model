# S-HAG: Scaled Heterogeneous Amdahl–Gustafson Model — Code and Data

Reproducibility package for the paper

> D. A. M. Hassan and K. I. Abu Zanouna, "Evaluating the Limits of Amdahl's Law in Heterogeneous Massively Parallel Architectures: A Gustafson-Inspired Reformulation," Computer Engineering Department, Faculty of Engineering, Islamic University of Gaza, 2026.

The repository has one executable notebook, `SHAG_complete.ipynb`. It holds the S-HAG reference implementation, the consolidated scaling dataset, the full analysis behind Section VI of the paper, and the cluster measurement kit. Running it from top to bottom regenerates every table value and figure reported in the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/dahassanm-jpg/shag-speedup-model/blob/main/SHAG_complete.ipynb)

**DOI:** `<ZENODO-DOI>`

---

## Quick start

**Google Colab (recommended, no installation)**
1. Click the *Open in Colab* badge above, or upload `SHAG_complete.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Select **Runtime → Run all**.
3. The last main cell (Section 10) creates `SHAG_outputs.zip`, which holds all results and figures, and downloads it to your computer.

**Local machine**
```bash
pip install numpy scipy pandas matplotlib jupyter
jupyter notebook SHAG_complete.ipynb      # then: Run All
```
The notebook writes everything into a `shag_work/` folder next to itself. It needs an internet connection the first time it runs, to fetch the published source data (see *Data*).

Python 3.9 or later is enough. The notebook prints the versions of Python, NumPy, pandas, and Matplotlib it ran with. A GPU is not needed.

---

## Notebook contents

| Section | What it does | Paper section |
|---|---|---|
| 1. Setup | Imports and the working-folder layout | — |
| 2. Reference implementation | Writes `model/shag.py`: G(P), D(P), the capacity terms H and Ψ, the communication term τ(P), the variability term m(P) (Blom approximation), S̄(P), E_H, the closed-form optimum P\* (18), the scaled limit (19), and the calibration routines | IV |
| 3. Model checks | Reduces S-HAG to Amdahl, Gustafson, Sun–Ni, Hill–Marty, and USL; runs a synthetic calibrate-small / predict-large test; draws Fig. 1 and the parametric results | IV-C, IV-D |
| 4. Data | Downloads the LAMMPS and BERT source files from fixed commits and checks each checksum; writes the transcribed GROMACS, ResNet-50, MPAS-A, and CESM tables | V-B, V-C |
| 5. Dataset build | Builds `scaling_dataset.csv` (218 configurations, 33 series) and runs the 27 consistency checks | V-D |
| 6. Validation E1–E4 | Held-out extrapolation (Amdahl, Gustafson, USL, S-HAG-B, S-HAG-A, S-HAG-A\*), GROMACS CPU→GPU transfer, Amdahl applied to weak scaling, and the location of the optimum | VI-A – VI-C |
| 7. Refinement tests E5 | Shrinking halo term, FFT work growth, and topology-aware term (exploratory) | VI-C |
| 8. Figures | Overview figures and paper Figs. 2–4 | VI |
| 9. Summary tables | Prints the medians and counts behind Tables V–X | VI |
| 10. Download | Packs `results/` and `figures/` into `SHAG_outputs.zip` | — |
| Appendix A | Cluster measurement kit for new runs (writes files only; it does not run here) | V-D |

---

## Data

All results come from **published measurements**. No new experiments were run for the paper.

| Code | Source | How it was obtained |
|---|---|---|
| LAMMPS | Thareja reproduction study [18] (FABRIC testbed) | CSV downloaded from `kthare10/lammps-reproducibility`, commit `d8f9726…`, MD5-checked |
| GROMACS | Kutzner et al. [20] | Transcribed from the published tables |
| ResNet-50 | Mikami et al. [21] (ABCI) | Transcribed from the published table |
| BERT-large | NVIDIA DeepLearningExamples [22] | Performance tables parsed from the README files at commit `729963d…`, MD5-checked |
| MPAS-A | Heinzeller et al. [23], Tables D1 and 3 | Transcribed |
| CESM | Kliphuis et al. [24] | Transcribed |

The first line of every transcribed CSV names its source table. If a downloaded file ever differs from the analysed version, the notebook stops with a checksum error instead of producing different numbers silently.

---

## Outputs

After a full run, `shag_work/` contains the following.

**`results/`**
- `scaling_dataset.csv`: the consolidated dataset
- `consistency_checks.csv`: the 27 transcription checks
- `per_series_accuracy.csv`: MAPE and signed error per series and model (Tables V, VI, X)
- `predictions.csv`: predicted vs. observed speedup at every held-out configuration
- `fitted_parameters.json`: fitted parameters of every model and series
- `heterogeneity_gromacs.csv`: CPU→GPU transfer test (Table VIII)
- `amdahl_weak_underestimation.csv`: Amdahl applied to weak scaling (Table VII)
- `improvements_test.csv`, `improvement1_node_test.csv`: refinement tests (Table IX)
- `section4_model_checks.json`, `section4_synthetic_experiment.json`: Section IV checks

**`figures/`**
- `fig1_shag.png`, `fig2_lammps.png`, `fig3_gromacs.png`, `fig4_cases.png`, `error_summary.png`

The analysis is deterministic. Repeated runs, including an independent run on Google Colab, reproduced every reported value to within 10⁻⁷.

---

## Cluster measurement kit (Appendix A)

Appendix A writes a Slurm-based kit to `shag_work/kit/`. The kit collects the component timings that the calibration protocol (Section IV-E) needs on a real cluster:

- `md/`: LAMMPS and GROMACS scaling jobs
- `dl/`: PyTorch DDP benchmark (per-step compute, optimizer, and all-reduce times)
- `climate/`: MPAS scaling job and a CESM PE-layout sweep guide
- `microbench/`: OSU Micro-Benchmarks job and `fwq.c`, a fixed-work-quantum noise benchmark (C/MPI)
- `common/`, `parsers/`, `collect.py`, `analyze_own_runs.py`: metadata capture, log parsing, and evaluation of your own runs with designs E1–E4

See `kit/README.md` for instructions. The paper does not report runs of this kit yet; it describes the protocol for future measurements.

---

## Citation

If you use this code or dataset, please cite the paper above and this archive:

```
@misc{hassan2026shag_code,
  author = {Hassan, Dalya Awad Mohammad and Abu Zanouna, Khalil Ibrahim},
  title  = {S-HAG: Scaled Heterogeneous Amdahl--Gustafson Model -- Code and Data},
  year   = {2026},
  doi    = {<ZENODO-DOI>},
  url    = {https://github.com/dahassanm-jpg/shag-speedup-model}
}
```

## License

Code: `<LICENSE, e.g. MIT>`. The transcribed data values come from the cited publications and remain attributed to their original authors; please cite those sources when you reuse them.
