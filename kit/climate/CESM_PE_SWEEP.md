# CESM scaling sweep (B1850, f19_g17) with the CIME timing files

CESM is a coupled model: each component (ATM, OCN, LND, ICE, CPL) runs on its own set of
processing elements (PEs). This is the heterogeneity S-HAG represents with phase capacities H_k (eq. (3)).

1. `./create_newcase --case b1850_pe$N --compset B1850 --res f19_g17 --machine <yours>`
2. For each total PE count N in {128, 256, 512, 1024, 2048} set the layout with `./xmlchange NTASKS=$N`
   (all components sequential on all PEs) **and** a second, concurrent layout where OCN has its own
   PEs (`NTASKS_OCN`, `ROOTPE_OCN`). Keep threads = 1.
3. `./xmlchange STOP_OPTION=nmonths,STOP_N=1,REST_OPTION=never,DOUT_S=FALSE` then build and submit.
   Repeat each layout 3 times.
4. Copy `timing/cesm_timing.*` into `results/cesm/pe$N/rep$r/`. `collect.py` reads model throughput
   (simulated years/day), per-component run times (`ATM Run Time`, `OCN Run Time`, …) and coupler
   communication time (`CPL COMM Time`).
5. The CIME load-balancing tool (`load_balancing_submit.py` / `load_balancing_solve.py`) can generate the
   sweep automatically; its timing files are compatible with `collect.py`.
