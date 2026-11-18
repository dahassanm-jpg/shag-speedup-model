"""Walk results/ (written by the job scripts) and build components.csv and microbench.csv for
analyze_own_runs.py.

  results/lammps/<mode>/<bench>/np<P>/rep<r>/log.lammps
  results/gromacs/<system>-<cpu|gpu>/nodes<N>/rep<r>/md.log
  results/dl/nodes<N>/{resnet50_weak,bert_strong}.jsonl
  results/mpas/<mesh>/np<P>/rep<r>/log.atmosphere.0000.out
  results/cesm/pe<P>/rep<r>/cesm_timing.*
  results/microbench/np<P>/{osu_allreduce,osu_alltoall,osu_barrier}_rep<r>.txt, fwq_rep<r>.txt

Usage: python collect.py [results_dir] [--allreduce-bytes N] [--alltoall-bytes N]
The *-bytes options give the application's message size used to read the micro-benchmark curves
(e.g. gradient bytes for DL, FFT transpose block for PME/PPPM).
"""
import argparse, re
from pathlib import Path
import pandas as pd
from parsers import parsers as PR


def num_after(prefix, s):
    m = re.search(prefix + r"(\d+)", s)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--allreduce-bytes", type=int, default=None)
    ap.add_argument("--alltoall-bytes", type=int, default=None)
    a = ap.parse_args()
    R = Path(a.results)
    rows = []
    for f in R.glob("lammps/*/*/np*/rep*/log.lammps"):
        mode, bench = f.parts[-5], f.parts[-4]
        d = PR.parse_lammps(f)
        rows.append(dict(app="LAMMPS", series=f"LAMMPS-{bench}-{mode}", mode=mode,
                         rep=num_after("rep", f.parts[-2]), source_file=str(f), **d))
    for f in R.glob("gromacs/*/nodes*/rep*/md.log"):
        d = PR.parse_gromacs(f)
        d.pop("sections", None)
        rows.append(dict(app="GROMACS", series=f"GROMACS-{f.parts[-4]}", mode="strong",
                         P=num_after("nodes", f.parts[-3]), rep=num_after("rep", f.parts[-2]),
                         source_file=str(f), **d))
    for f in R.glob("dl/nodes*/*.jsonl"):
        for i, d in enumerate(PR.parse_ddp(f)):
            rows.append(dict(app=d.pop("model"), series=f"DL-{f.stem}", rep=i + 1, source_file=str(f), **d))
    for f in R.glob("mpas/*/np*/rep*/log.atmosphere.0000.out"):
        d = PR.parse_mpas(f)
        d.pop("timers", None)
        rows.append(dict(app="MPAS-A", series=f"MPAS-{f.parts[-4]}", mode="strong",
                         P=num_after("np", f.parts[-3]), rep=num_after("rep", f.parts[-2]), source_file=str(f), **d))
    for f in R.glob("cesm/pe*/rep*/cesm_timing*"):
        d = PR.parse_cesm(f)
        comp = d.pop("component_run_times")
        rows.append(dict(app="CESM", series="CESM", mode="strong", rep=num_after("rep", f.parts[-2]),
                         t_serial=0.0, t_parallel=d["t_total"] - d["t_comm"], t_imbalance=0.0,
                         source_file=str(f), **d, **{f"t_{k.lower()}": v for k, v in comp.items()}))
    C = pd.DataFrame(rows)
    C.to_csv("components.csv", index=False)
    print(f"components.csv: {len(C)} runs in {C.series.nunique() if len(C) else 0} series")

    mrows = []
    for f in R.glob("microbench/np*/osu_*_rep*.txt"):
        kind = re.search(r"osu_(\w+?)_rep", f.name).group(1)
        curve = PR.parse_osu(f)
        size = {"allreduce": a.allreduce_bytes, "alltoall": a.alltoall_bytes}.get(kind)
        lat = curve[min(curve, key=lambda s: abs(s - size))] if (curve and size) else (curve.get(0) or next(iter(curve.values()), None))
        mrows.append(dict(P=num_after("np", f.parts[-2]), kind=kind, rep=num_after("rep", f.name),
                          latency_us=lat, bytes=size))
    for f in R.glob("microbench/np*/fwq_rep*.txt"):
        d = PR.parse_fwq(f)
        mrows.append(dict(P=d["P"], kind="fwq", rep=num_after("rep", f.name), sigma_a_s=d["sigma_a"]))
    if mrows:
        pd.DataFrame(mrows).to_csv("microbench.csv", index=False)
        print(f"microbench.csv: {len(mrows)} rows. Map them to an application series (column 'series',"
              " per-step time t_comm_micro = latency x communications per run) before passing to"
              " analyze_own_runs.py")


if __name__ == "__main__":
    main()
