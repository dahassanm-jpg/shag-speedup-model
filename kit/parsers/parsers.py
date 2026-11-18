"""Log parsers that turn application output into the component schema used by analyze_own_runs.py.

Component schema (one row per run):
  app, series, mode, P, rep, t_total, t_serial, t_parallel, t_comm, t_imbalance, source_file
    t_serial    time not reduced by more resources (I/O, setup inside the timed loop, replicated work)
    t_parallel  time of divisible computation (average over ranks)
    t_comm      communication + synchronisation time (average over ranks)
    t_imbalance max-minus-average of the parallel work (load imbalance / jitter at barriers)

Status of each parser (see README):
  parse_lammps   tested on a real LAMMPS log (stable 2Aug2023+ 'MPI task timing breakdown')
  parse_ddp      tested on real ddp_bench.py output (CPU/gloo)
  parse_fwq      tested on real fwq output
  parse_osu      written for the OSU Micro-Benchmarks 7.x text format
  parse_gromacs  written for the GROMACS 2020+ md.log cycle-accounting table
  parse_mpas     written for the MPAS-A log.atmosphere timer table
  parse_cesm     written for the CIME timing file (cesm_timing.*)
The last four have not been run on real output here: check the first parsed file by hand.
"""
import json, re
from pathlib import Path
import pandas as pd

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_lammps(path):
    t = Path(path).read_text(errors="replace")
    loops = list(re.finditer(rf"Loop time of {NUM} on (\d+) procs for (\d+) steps with (\d+) atoms", t))
    if not loops:
        raise ValueError(f"no 'Loop time' line in {path}")
    m = loops[-1]                                   # last run block = production run
    rest = t[m.end():]
    sec = {}
    for name in ("Pair", "Bond", "Kspace", "Neigh", "Comm", "Output", "Modify", "Other"):
        r = re.search(rf"^{name}\s*\|\s*(?:{NUM})?\s*\|\s*{NUM}\s*\|\s*(?:{NUM})?\s*\|", rest, re.M)
        if r:
            mn, avg, mx = r.group(1), r.group(2), None
            row = re.search(rf"^{name}\s*\|([^\n]*)", rest, re.M).group(1).split("|")
            vals = [v.strip() for v in row]
            sec[name.lower()] = dict(min=float(vals[0]) if vals[0] else None, avg=float(vals[1]),
                                     max=float(vals[2]) if vals[2] else None)
    g = lambda k, f="avg": (sec.get(k, {}).get(f) or 0.0)
    par = g("pair") + g("bond") + g("neigh") + g("modify")
    imb = (g("pair", "max") - g("pair")) if sec.get("pair", {}).get("max") else 0.0
    return dict(P=int(m.group(2)), steps=int(m.group(3)), natoms=int(m.group(4)), t_total=float(m.group(1)),
                t_serial=g("output") + g("other"), t_parallel=par, t_comm=g("comm"),
                t_kspace=g("kspace"), t_imbalance=imb)


def parse_ddp(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        rows.append(dict(P=d["world"], t_total=d["t_step_mean"], t_serial=d["t_opt_mean"],
                         t_parallel=d["t_compute_mean"],
                         t_comm=max(d["t_step_mean"] - d["t_compute_mean"] - d["t_opt_mean"], 0.0),
                         t_comm_micro=d["t_allreduce_mean"],
                         t_imbalance=d["t_compute_max_over_ranks"] - d["t_compute_mean"],
                         global_batch=d["global_batch"], mode=d["mode"], model=d["model"]))
    return rows


def parse_fwq(path):
    d = pd.read_csv(path, comment="#")
    return dict(P=len(d), sigma_a=float(d.sd_s.mean()), mean_quantum=float(d.mean_s.mean()),
                max_over_ranks=float(d.max_s.max()))


def parse_osu(path):
    """Returns {message_size_bytes: avg_latency_us}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            out[int(parts[0])] = float(parts[1])
    return out


def parse_gromacs(path):
    t = Path(path).read_text(errors="replace")
    perf = re.search(rf"Performance:\s+{NUM}\s+{NUM}", t)
    wall = re.search(rf"Time:\s+{NUM}\s+{NUM}", t)
    acc = t[t.find("R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G"):]
    rows = {}
    for line in acc.splitlines():
        r = re.match(rf"\s*([A-Za-z][A-Za-z .\-/()]+?)\s{{2,}}(\d+)\s+(\d+)\s+(\d+)?\s*{NUM}\s+{NUM}\s+{NUM}\s*$", line)
        if r:
            rows[r.group(1).strip()] = float(r.group(6))     # wall time (s)
    comm_keys = [k for k in rows if k.startswith(("Comm.", "Wait", "PME redist", "PME 3D-FFT Comm", "Domain decomp"))]
    serial_keys = [k for k in rows if k.startswith(("Write traj", "Update", "Constraints", "Rest"))]
    total = rows.get("Total", float(wall.group(2)) if wall else float("nan"))
    t_comm = sum(rows[k] for k in comm_keys)
    t_ser = sum(rows[k] for k in serial_keys)
    return dict(ns_per_day=float(perf.group(1)) if perf else None, t_total=total, t_comm=t_comm,
                t_serial=t_ser, t_parallel=max(total - t_comm - t_ser, 0.0), t_imbalance=rows.get("Wait GPU NB local", 0.0),
                sections=rows)


def parse_mpas(path):
    """MPAS-A log.atmosphere.0000.out timer table: '  N  name   total  calls  min  max  avg ...'"""
    t = Path(path).read_text(errors="replace")
    tim = {}
    for line in t.splitlines():
        r = re.match(rf"\s*\d+\s+([a-zA-Z_][\w ]*?)\s+{NUM}\s+(\d+)\s+{NUM}\s+{NUM}\s+{NUM}", line)
        if r:
            tim.setdefault(r.group(1).strip(), dict(total=float(r.group(2)), min=float(r.group(4)),
                                                     max=float(r.group(5)), avg=float(r.group(6))))
    g = lambda k: tim.get(k, {}).get("avg", 0.0)
    total = tim.get("total time", {}).get("total", float("nan"))
    comm = sum(v["avg"] for k, v in tim.items() if "halo" in k or "exchange" in k or "comm" in k)
    serial = g("initialize") + sum(v["avg"] for k, v in tim.items() if "write" in k or "output" in k)
    integ = tim.get("time integration", {})
    imb = (integ.get("max", 0) - integ.get("avg", 0)) if integ else 0.0
    return dict(t_total=total, t_serial=serial, t_comm=comm, t_parallel=max(total - serial - comm, 0.0),
                t_imbalance=imb, timers=tim)


def parse_cesm(path):
    t = Path(path).read_text(errors="replace")
    tp = re.search(rf"Model Throughput:\s+{NUM}\s+simulated_years/day", t)
    pes = re.search(r"total pes active\s*:\s*(\d+)", t)
    comp = {m.group(1): float(m.group(2)) for m in
            re.finditer(rf"^\s*([A-Z]{{3}}) Run Time:\s+{NUM}\s+seconds", t, re.M)}
    comm = sum(float(m.group(1)) for m in re.finditer(rf"CPL COMM Time:\s+{NUM}", t))
    tot = comp.get("TOT", float("nan"))
    return dict(sypd=float(tp.group(1)) if tp else None, P=int(pes.group(1)) if pes else None,
                t_total=tot, component_run_times=comp, t_comm=comm)
