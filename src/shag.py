"""Reference implementation of the revised S-HAG model (Section IV).

Mean-time speedup:
    S(P) = G(P) / D(P)
    G(P) = s0*gs(P) + (1-s0)*gp(P)                     (scaled reference work)
    D(P) = s0*gs(P)/rho_s + (1-s0)*sum_k f_k*gp(P)/H_k(P) * [1 + cv*m(P)] + tau(P) + zeta*m(P)
with
    tau(P) = tau0 + lam*log2(P) + beta*(P-1)            (communication)
    m(P) = E[max of P std normals]; cv = per-worker compute jitter (coefficient of variation),
    zeta = k*sigma_a/T1 additive (OS-noise) jitter accumulated over k synchronisations
All times normalised by T1(W0).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares, brentq


def m_max(P):
    """Expected maximum of P i.i.d. standard normals (Blom approximation); m(1)=0."""
    P = np.asarray(P, dtype=float)
    out = norm.ppf((P - 0.375) / (P + 0.25))
    return np.where(P <= 1, 0.0, out)


def m_max_exact(P):
    """Numerical E[max of P std normals] for checking the approximation."""
    from scipy.integrate import quad
    if P <= 1:
        return 0.0
    f = lambda x: x * P * norm.pdf(x) * norm.cdf(x) ** (P - 1)
    return quad(f, -12, 12, limit=200)[0]


def tau(P, tau0=0.0, lam=0.0, beta=0.0, reference_alone=True):
    """Eq. (4). tau = 0 only for the reference resource executing alone (P = 1 and
    reference_alone=True); any other configuration, including a single non-reference
    device that needs host-device transfers, gets tau0 + lam*log2 P + beta*(P-1)."""
    P = np.asarray(P, dtype=float)
    val = tau0 + lam * np.log2(P) + beta * (P - 1)
    return np.where((P <= 1) & reference_alone, 0.0, val)


def speedup(P, s0, H=None, gs=None, gp=None, rho_s=1.0, phases=None,
            tau0=0.0, lam=0.0, beta=0.0, zeta=0.0, cv=0.0, tau_fn=None):
    """Predicted mean-time speedup. H, gs, gp are callables of P (defaults: P, 1, 1).
    phases: optional list of (f_k, H_k callable) with sum f_k = 1."""
    P = np.asarray(P, dtype=float)
    H = H or (lambda p: p)
    gs = gs or (lambda p: np.ones_like(p))
    gp = gp or (lambda p: np.ones_like(p))
    G = s0 * gs(P) + (1 - s0) * gp(P)
    if phases is None:
        par = (1 - s0) * gp(P) / H(P)
    else:
        par = (1 - s0) * gp(P) * sum(f / Hk(P) for f, Hk in phases)
    t = tau_fn(P) if tau_fn else tau(P, tau0, lam, beta)
    mP = m_max(P)
    D = s0 * gs(P) / rho_s + par * (1 + cv * mP) + t + zeta * mP
    return G / D


def amdahl(P, s):
    return 1.0 / (s + (1 - s) / P)


def gustafson(P, a):
    return P - a * (P - 1)


def usl(P, sigma, kappa):
    return P / (1 + sigma * (P - 1) + kappa * P * (P - 1))


def karp_flatt(S, P):
    return (1 / S - 1 / P) / (1 - 1 / P)


def p_star_log(s, lam):
    """Optimal P for H=P, tau=lam*ln P (natural log)."""
    return (1 - s) / lam


def p_star_lin(s, beta):
    """Optimal P for H=P, tau=beta*(P-1)."""
    return np.sqrt((1 - s) / beta)


def fit(P, S_obs, free=("s0", "lam", "beta", "zeta"), fixed=None, H=None):
    """Least-squares calibration of S-HAG on log-speedup (relative error)."""
    fixed = dict(fixed or {})
    x0 = {"s0": 0.05, "lam": 1e-3, "beta": 1e-5, "zeta": 1e-3, "tau0": 0.0}
    lo = {"s0": 0.0, "lam": 0.0, "beta": 0.0, "zeta": 0.0, "tau0": 0.0}
    hi = {"s0": 1.0, "lam": 1.0, "beta": 1.0, "zeta": 1.0, "tau0": 1.0}

    def pred(x, PP):
        kw = dict(fixed); kw.update(dict(zip(free, x)))
        return speedup(PP, H=H, **kw)

    r = least_squares(lambda x: np.log(pred(x, P)) - np.log(S_obs),
                      [x0[k] for k in free], bounds=([lo[k] for k in free], [hi[k] for k in free]))
    params = dict(zip(free, r.x)); params.update(fixed)
    return params, (lambda PP: pred(r.x, PP))
