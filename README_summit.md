Building and running HiOp on `summit` is quite simple since the dependencies of HiOp can be satisfied quickly using the `module` utility.
```
module load gcc/8.1.1 cmake/3.17.3 netlib-lapack/3.8.0 cuda/10.1.105 magma/2.5.1
```
Different versions of the above modules may conflict with each other; in this case, one needs to inquire additional compatibility information using `module spider package_name`.

Next HiOp with GPU support can be `cmake`'d, built, and tested using
```
cd build/
CC=/sw/summit/gcc/8.1.1/bin/gcc CXX=/sw/summit/gcc/8.1.1/bin/g++ cmake -DHIOP_USE_GPU=ON ..
make -j
make test
make install
```
In some cases, `cmake` picks up a different gcc compiler than the one loaded via `module`. This situation can be remedied by preloading the `CC` and `CXX` for `cmake` as it is done above.

Depending on the LAPACK/BLAS CPU library loaded by the `module` command, runtime crashes can occur (intermintently for different problems/linear systems sizes) if the `nvblas.conf` cannot be find or is not specifying the so-called NVBLAS CPU BLAS library required by CUDA. This can be remedied by specifying the location of the BLAS library in `NVBLAS_CPU_BLAS_LIB` in the `nvblas.conf` file; for example, this can be done by
```
NVBLAS_CPU_BLAS_LIB /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-lapack-3.8.0-p74bsneivus4jck562lq7drw2s7i4ytd/lib64/libblas.so
```
For non-standard locations of `nvblas.conf`, one can specify the path to `nvblas.conf` via the environment variable `NVBLAS_CONFIG_FILE`; for example, this can be achieved by exporting the variable in your submission `your_job_file.lsf` file:
```
# ##LSF directives  here
#  other commands here
# example assumes a bash shell
export NVBLAS_CONFIG_FILE=/ccs/home/cpetra/work/projects/hiop/runs/nvblas.conf 

# jsrun command here

```