/* Fixed-Work-Quantum (FWQ) noise benchmark (Section IV-E step 5).
 * Every rank repeatedly times an identical amount of work; the spread of the samples is the
 * operating-system/hardware noise sigma_a. Output: one line per rank with mean, sd, max (seconds)
 * and a CSV of all samples if a file name is given.
 * Build: mpicc -O1 -o fwq fwq.c        Run: mpirun -np P ./fwq [n_samples] [work] [samples.csv]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double work(long n) {             /* volatile-free dependent chain the compiler keeps */
    double x = 1.0;
    for (long i = 0; i < n; i++) x = x * 1.0000001 + 1e-9;
    return x;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    int ns = argc > 1 ? atoi(argv[1]) : 2000;
    long nw = argc > 2 ? atol(argv[2]) : 200000;
    double *t = malloc(sizeof(double) * ns), sink = 0;
    for (int w = 0; w < 50; w++) sink += work(nw);          /* warm-up */
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < ns; i++) { double t0 = MPI_Wtime(); sink += work(nw); t[i] = MPI_Wtime() - t0; }
    double m = 0, s = 0, mx = 0;
    for (int i = 0; i < ns; i++) { m += t[i]; if (t[i] > mx) mx = t[i]; }
    m /= ns; for (int i = 0; i < ns; i++) s += (t[i] - m) * (t[i] - m); s = sqrt(s / (ns - 1));
    double *all = rank == 0 ? malloc(sizeof(double) * ns * size) : NULL;
    MPI_Gather(t, ns, MPI_DOUBLE, all, ns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double stats[3] = {m, s, mx}, *st = rank == 0 ? malloc(sizeof(double) * 3 * size) : NULL;
    MPI_Gather(stats, 3, MPI_DOUBLE, st, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("# FWQ ranks=%d samples=%d work=%ld (sink=%g)\nrank,mean_s,sd_s,max_s\n", size, ns, nw, sink);
        for (int r = 0; r < size; r++) printf("%d,%.9e,%.9e,%.9e\n", r, st[3*r], st[3*r+1], st[3*r+2]);
        if (argc > 3) { FILE *f = fopen(argv[3], "w"); fprintf(f, "rank,sample,t_s\n");
            for (int r = 0; r < size; r++) for (int i = 0; i < ns; i++) fprintf(f, "%d,%d,%.9e\n", r, i, all[r*ns+i]);
            fclose(f); }
    }
    MPI_Finalize(); return 0;
}
