# Wavelet Image Compression

Parallelize the one-dimensional quantum dynamics (QD) simulation program, by combining MPI and CUDA technologies. Tested on USC HPC cluster.

## File description

* qd1.c - Single thread quantum dynamics simulations
* mpqd1.c - MPI parallelized quantum dynamics simulations
* mpcuqd1.cu - MPI parallelized quantum dynamics simulations with energy calculations handled by GPU through CUDA

## Running the tests

```
mpicc mpqd1.c -o mpcuqd1 -lm
nvcc -I/usr/usc/openmpi/default/include -L/usr/usc/openmpi/default/lib -lmpi mpcuqd1.cu -o mpcuqd1
qsub mpcuqd1.pbs
```

## Simulation system
L_x = 250 au each rank
N_x = 2496 au
DeltaT * Nstep = 50 au total simulated
2e-3 au * 2.5e4 = 1.4e-17 sec
E0 = k0^2/2 = 100 au

|| E_h = 50 au                           ||
||                       Barrier         ||
||                  |-|  B w = 20 au     ||
||                  | |  B_h = 5 au      ||
||                  | |                  ||
||---------|---------|---------|---------||
   rank 0       1         2         3
