# Wavelet Image Compression

Perform image compression using wavelets through a hybrid MPI+OpenMP program. Tested with Lenna greyscale image on USC HPC cluster.

## File description

* imagerw.c - Image read and write functions
* wavelet.c - Single thread wavelet image compression algorithm
* mpwavelet.c - MPI parallelized wavelet image compression algorithm

## Running the tests

```
qsub wavelet.pbs
```

