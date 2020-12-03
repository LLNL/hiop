Building and running HiOp on `summit` is quite simple since the dependencies of HiOp can be satisfied quickly using the `module` utility.
```
export INSTALL_DIR=$PROJWORK/csc359/installs/ExaSGD

module purge
export EXASGD_ROOT=$INSTALL_DIR
module use $INSTALL_DIR/Modulefiles/Core

#Then a project-specific compiler module so you can access                                                                                                                            
#software built with that compiler.                                                                                                                                                   

module load gcc-esgd/9.2.0

#Then a project-specific MPI module so you can access                                                                                                                                 
#software built with that compiler and MPI library.                                                                                                                                   

module load spectrum-mpi-esgd

module load cuda/11.0.1
module load mpfr/4.1.0
module load openblas/0.3.10
module load hdf5/1.10.4

module load hwloc/2.0.2-py3
module load cmake/3.13.4
module load umpire/4.0.1-cuda11
module load magma/master-9ce41caa-cuda11
module load metis/5.1.0
module load raja/0.12.1-cuda11
module load suitesparse/5.8.1-cuda11

module load hiop/raja-snapshot20200920-dev-31aa10c-cuda11
module load petsc/3.13.5
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

Developers are usually required to perform a more comprehensive build
```
rm -rf *
CC=/sw/summit/gcc/9.2.0/bin/gcc CXX=/sw/summit/gcc/9.2.0/bin/g++ cmake -DHIOP_SPARSE=ON -DHIOP_USE_GPU=ON -HIOP_TEST_WITH_BSUB=ON -DHIOP_USE_MPI=ON -DMETIS_DIR=$INSTALL_DIR/Compiler/gcc-9.2.0/metis/5.1.0/ -DHIOP_COINHSL_DIR=/ccs/home/cpetra/work/installs/coinhsl-2015.06.23/_install -DCMAKE_BUILD_TYPE=DEBUG -DHIOP_TEST_WITH_BSUB=ON -DHIOP_USE_RAJA=ON -DHIOP_USE_UMPIRE=ON -DHIOP_USE_GPU=ON -DHIOP_DEEPCHECKS=ON .. 
make -j

Depending on the LAPACK/BLAS CPU library loaded by the `module` command, runtime crashes can occur (intermintently for different problems/linear systems sizes) if the `nvblas.conf` cannot be find or is not specifying the so-called NVBLAS CPU BLAS library required by CUDA. This can be remedied by specifying the location of the BLAS library in `NVBLAS_CPU_BLAS_LIB` in the `nvblas.conf` file; for example, this can be done by adding the following lines to `nvblas.conf` file 
```
NVBLAS_CPU_BLAS_LIB /gpfs/alpine/proj-shared/csc359/installs/ExaSGD/Compiler/gcc-9.2.0/openblas/0.3.10/lib/libopenblas.so
```
Be aware that this assumes that the openblas has been `module load`ed

For non-standard locations of `nvblas.conf`, one can specify the path to `nvblas.conf` via the environment variable `NVBLAS_CONFIG_FILE`; for example, this can be achieved by exporting the variable in your submission `your_job_file.lsf` file:
```
# ##LSF directives  here
#  other commands here
# example assumes a bash shell
export NVBLAS_CONFIG_FILE=/ccs/home/cpetra/work/projects/hiop/runs/nvblas.conf 

# jsrun command here

```