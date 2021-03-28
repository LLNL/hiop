# Option 1: using libraries generally available on Summit

Building and running HiOp on Summit is quite simple since the dependencies of HiOp can be satisfied quickly using the `module` utility. Please note that on Summit only a limited number of dependecies are available by default and, as a result, HiOp cannot be usually built with all the features (for example dependencies for sparse linear algebra will not be satisfied).
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

# Option 2 - `module load` precompiled dependencies made available by exaSGD project 

The commands below will satisfy all the depedencies needed to build HiOp with all possible features. However, some of the modules listed below are updated regularily and their names may change.  

```bash
#!/bin/bash

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
```

# Option 3 - using Spack

## Using official Spack recipe
An Spack recipe is being developed and will be available soon.

## Custom Spack-based build to build HiOp's sparse

#### Method 1:
Install hiop and all its dependencies from scratch via spack:
1.	Download ExaSGD_Spack repository 
```bash
git clone ssh://git@gitlab.pnnl.gov:2222/exasgd/ExaSGD_Spack.git 
```
2.	 Switch to branch summit-install 
```bash
cd ExaSGD_spack
git checkout summit-install
```
3.	Install spack
```bash
./install.sh summit
```
4.	Source spack environment file. This should allow you to run spack commands.
```bash
source spack/share/spack/setup-env.sh
```
5.	Load the precompiled modules used by the spack exago script, which is in folder hiop/scripts under HiOp's root directory. 
```
#make sure you are in hiop/scripts under HiOp's root directory
source build_summit.sh
```
6.	Create spack environment, using `environments/summit/hiop-develop-sparse.yaml`. The yaml file has configuration for installing hiop and all its dependencies. 
            (To build ExaGO, please use other yaml files in the folder environments/summit)
```bash
spack env create hiop-sparse-coinhsl-llnl environments/summit/hiop-develop-sparse.yaml
```
7.	Activate the environment
```bash
spack env activate hiop-sparse-coinhsl-llnl
```
8.	Download coinhsl-2015.06.23.tar.gz from hsl, and rename it as coinhsl-archive-2015.06.23.tar.gz. Place this file in the current folder (ExaSGD_spack) 
9.	Install the packages
```bash
spack install
```

#### Method 2:
Install *ONLY* coinhsl via spack:
1.	Download ExaSGD_Spack repository 
```bash
git clone ssh://git@gitlab.pnnl.gov:2222/exasgd/ExaSGD_Spack.git 
```
2.	 Switch to branch summit-install 
```bash
cd ExaSGD_spack
git checkout summit-install
```
3.	Install spack
```bash
./install.sh summit
```
4.	Source spack environment file. This should allow you to run spack commands.
```bash
source spack/share/spack/setup-env.sh
```
5.	Load some precompiled modules used by the project ExaGO. It is saved in folder hiop/scripts. (in branch summit-env-dev)
```bash
source build_summit.sh 
```
6.	Download coinhsl-2015.06.23.tar.gz from hsl, and rename it as coinhsl-archive-2015.06.23.tar.gz. Place this file in the current folder (ExaSGD_spack) 
7.	Install the coinhsl, using existing gcc9.2.0 and openblas from project ExaGO
```bash
spack install coinhsl@2015.06.23%gcc@9.2.0+blas ^openblas@0.3.10
```


Once HiOp and coinhsl are installed, one can find the corresponding module files if needed:
1.	Find the installation path of coinhsl
```bash
spack location -i coinhsl
```
 For example, it returns 
```bash
/autofs/nccs-svm1_home1/chiang7/spack_install/ExaSGD_Spack/spack/opt/spack/linux-rhel7-power9le/gcc-9.2.0/coinhsl-2015.06.23-kizzx7n6vuwytfq7imrses6xidtpiruz
```
2.	Export this pathas below since another script will load coinhsl from this path later. NOTE the difference between the installation path and the module path!
```bash
export COINHSL_MODULE_DIR=/autofs/nccs-svm1_home1/chiang7/spack_install/ExaSGD_Spack/spack/share/spack/modules/linux-rhel7-power9le
```
#### The actual build (after all steps of Method 1 or Method 2 above were performed)
Finally one can build HiOp with all the dependencies (magma, metis, etc.) and coinhsl from the user’s Spack path.
To build HiOp with default options, one should issue the command
```bash
MY_CLUSTER=summit ./BUILD.sh --build-only
```
To build a customized HiOp, one can issue from the root folder of HiOp the following:
```bash
export BUILDDIR=$PWD/build
source scripts/summitVariables.sh
cd build
#and, for example, build the sparse solver of HiOp
cmake -DHIOP_SPARSE=ON <other_hiop_build_options>  ..
```
