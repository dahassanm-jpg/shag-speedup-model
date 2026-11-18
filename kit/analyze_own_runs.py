"""Component-calibrated S-HAG prediction for your own runs (Section V-D, protocol for new measurements).

Input : components.csv produced by collect.py (schema in parsers/parsers.py), optionally
        microbench.csv (P, t_comm_micro) from collect.py --microbench.
Output: results/own_predictions.csv, results/own_accuracy.csv

Per series:
  calibration P  = the ceil(n/2) smallest resource counts (>= 3); held-out = the rest
  T_ser          = t_serial at the smallest P (g_s = 1)
  T_par          = t_parallel at the smallest P, scaled by P0/P (strong) or constant (weak)
  c_v            = mean over calibration P of t_imbalance / (t_parallel * m(P))
  comm(P)        = nonnegative fit c0 + lam log2 P + beta (P-1) to t_comm at calibration P,
                   or to the micro-benchmark times at ALL target P when microbench.csv is given
                   (Section IV-E step 4: micro-benchmarks are cheap at full scale)
  kspace(P)      = (LAMMPS PPPM only) a/P + b log2 P + c (P-1) fitted at calibration P
  T(P)           = T_ser + T_par(P) [1 + c_v m(P)] + kspace(P) + comm(P)            -> eq. (7)
Classical baselines (Amdahl, Gustafson for weak, USL) are fitted to end-to-end speedups at the
same calibration P.  Usage:  python analyze_own_runs.py components.csv [microbench.csv]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import shag as M  # noqa: E402


def fitnn(P, y, cols):
    X = np.c_[[c(P) for c in cols]].T
    coef, _ = nnls(X, y)
    return coef


def run(comp_csv, micro_csv=None, out_dir="results"):
    C = pd.read_csv(comp_csv)
    micro = pd.read_csv(micro_csv) if micro_csv else None
    Path(out_dir).mkdir(exist_ok=True)
    preds, acc = [], []
    for sid, s in C.groupby("series"):
        g = s.groupby("P").mean(numeric_only=True).sort_index()
        sd = s.groupby("P").t_total.std()
        P = g.index.values.astype(float)
        n = len(P)
        nc = max(3, int(np.ceil(n / 2)))
        if n <= nc:
            print(f"{sid}: {n} resource counts -> no held-out point, skipped"); continue
        weak = s["mode"].iloc[0] == "weak"
        P0 = P[0]
        p = P / P0
        T = g.t_total.values
        S_obs = (p * T[0] / T) if weak else (T[0] / T)
        cal = slice(0, nc)
        ser, par0 = g.t_serial.values[0], g.t_parallel.values[0]
        par = par0 * (np.ones_like(P) if weak else P0 / P)
        mP = M.m_max(p)
        ok = (mP[cal] > 0) & (g.t_parallel.values[cal] > 0)
        cv = float(np.clip(np.mean(g.t_imbalance.values[cal][ok] / (g.t_parallel.values[cal][ok] * mP[cal][ok])), 0, None)) if ok.any() else 0.0
        base = [lambda q: np.ones_like(q), lambda q: np.log2(q / P0), lambda q: q / P0 - 1]
        if micro is not None and sid in set(micro.series):
            mm = micro[micro.series == sid].groupby("P").t_comm_micro.mean()
            cc = fitnn(mm.index.values.astype(float), mm.values, base)
            src = "micro-benchmark"
        else:
            cc = fitnn(P[cal], g.t_comm.values[cal], base)
            src = "calibration timers"
        comm = sum(c * f(P) for c, f in zip(cc, base))
        ks = np.zeros_like(P)
        if "t_kspace" in g and g.t_kspace.fillna(0).values.any():
            kb = ([lambda q: P0 / q] if not weak else [lambda q: np.ones_like(q)]) + base[1:]
            kc = fitnn(P[cal], g.t_kspace.fillna(0).values[cal], kb)
            ks = sum(c * f(P) for c, f in zip(kc, kb))
        T_pred = ser + par * (1 + cv * mP) + ks + comm
        S_shag = (p * T_pred[0] / T_pred) if weak else (T_pred[0] / T_pred)
        # classical baselines on end-to-end calibration points
        lsq = lambda f, x0, lo, hi: least_squares(lambda x: np.log(f(x, p[cal])) - np.log(S_obs[cal]),
                                                  x0, bounds=(lo, hi)).x
        models = {"S-HAG (components)": S_shag}
        a = lsq(lambda x, q: M.amdahl(q, x[0]), [0.05], [0], [0.999]); models["Amdahl"] = M.amdahl(p, a[0])
        u = lsq(lambda x, q: M.usl(q, *x), [0.02, 1e-4], [0, 0], [1, 1]); models["USL"] = M.usl(p, *u)
        if weak:
            gu = lsq(lambda x, q: M.gustafson(q, x[0]), [0.05], [0], [1]); models["Gustafson"] = M.gustafson(p, gu[0])
        h = slice(nc, n)
        for mname, v in models.items():
            for i in range(n):
                preds.append(dict(series=sid, P=P[i], role="cal" if i < nc else "held-out", model=mname,
                                  S_obs=S_obs[i], S_pred=v[i], t_total_sd=sd.iloc[i]))
            ape = np.abs(v[h] - S_obs[h]) / S_obs[h] * 100
            acc.append(dict(series=sid, model=mname, n=n, n_cal=nc, comm_source=src, c_v=cv,
                            MAPE=ape.mean(), maxAPE=ape.max()))
    pd.DataFrame(preds).to_csv(Path(out_dir) / "own_predictions.csv", index=False)
    A = pd.DataFrame(acc)
    A.to_csv(Path(out_dir) / "own_accuracy.csv", index=False)
    return A


if __name__ == "__main__":
    A = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(A.pivot_table(index="series", columns="model", values="MAPE").round(1).to_string())
