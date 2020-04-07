

# HiOp - HPC solver for optimization
HiOp is an optimization solver for solving certain mathematical optimization problems expressed as nonlinear programming problems. HiOp is a lightweight HPC solver that leverages application's existing data parallelism to parallelize the optimization iterations by using specialized linear algebra kernels.

## Build/install instructions
HiOp uses a CMake-based build system. A standard build can be done by invoking in the 'build' directory the following 
```shell 
$> cmake ..
$> make 
$> make test
$> make install
```
This sequence will build HiOp and install the headers and the shared library in the directory '_dist-default-build' in HiOp's root directory.

The installation can be customized using the standard CMake options. For example, one can provide an alternative installation directory for HiOp by using 
```sh
$> cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/hiop ..'
```


### HiOp-specific build options
* Enable/disable MPI: *-DHIOP_USE_MPI=[ON/OFF]* (by default ON)
* GPU support: *-DHIOP_USE_GPU=ON*. MPI can be either off or on. For more build system options related to GPUs, see "Dependencies" section below.
* Additional checks and self-diagnostics inside HiOp meant to detect anormalities and help to detect bugs and/or troubleshoot problematic instances: *-DHIOP_DEEPCHECKS=[ON/OFF]* (by default ON). Disabling HIOP_DEEPCHECKS usually provides 30-40% execution speedup in HiOp. For full strength, it is recomended to use HIOP_DEEPCHECKS with debug builds. With non-debug builds, in particular the ones that disable the assert macro, HIOP_DEEPCHECKS does not perform all checks and, thus, may overlook potential issues.

For example:
```shell 
$> cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON ..
$> make 
$> make test
$> make install
```


### Other useful options to use with CMake
* *-DCMAKE_BUILD_TYPE=Release* will build the code with the optimization flags on
* *-DCMAKE_CXX_FLAGS="-O3"* will enable a high level of compiler code optimization

### Dependencies
HiOp requires LAPACK and BLAS. These dependencies are automatically detected by the build system. MPI is optional and by default enabled. To disable use cmake option '-DHIOP_USE_MPI=OFF'.

HiOp has some support for NVIDIA **GPU-based computations** via CUDA and Magma. To enable the use of GPUs,  use cmake with '-DHIOP_USE_GPU=ON'. The build system will automatically search for CUDA Toolkit. For non-standard CUDA Toolkit installations, use '-DHIOP_CUDA_LIB_DIR=/path' and '-DHIOP_CUDA_INCLUDE_DIR=/path'. For "very" non-standard CUDA Toolkit installations, one can specify the directory of cuBlas libraries as well with '-DHIOP_CUBLAS_LIB_DIR=/path'.

### Support for GPU computations

When GPU support is on, HiOp also requires the specifciation of the path to Magma installation directory using '-DHIOP_MAGMA_DIR=/path'. The typical cmake command to enable GPU support in HiOp is
```shell 
$> cmake -DHIOP_USE_GPU=ON -DHIOP_MAGMA_DIR=/home/petra1/work/installs/magma-2.5.2/lib ..
```
For custom CUDA Toolkit installations, an example of the cmake command would be
```shell 
$> cmake -DHIOP_USE_GPU=ON -DHIOP_MAGMA_DIR=/home/petra1/work/installs/magma-2.5.2/lib -DHIOP_CUDA_INCLUDE_DIR=/usr/local/cuda-10.2/include/ -DHIOP_CUDA_LIB_DIR=/usr/local/cuda-10.2/lib64 -DHIOP_CUBLAS_LIB_DIR=/usr/local/cuda-10.2/targets/x86_64-linux/lib/lib64 ..
```

### Kron reduction

Kron reduction functionality of HiOp is disabled by default. One can enable it by using 
```shell
$> rm -rf *; cmake -DHIOP_WITH_KRON_REDUCTION=ON -DUMFPACK_DIR=/Users/petra1/work/installs/SuiteSparse-5.7.1 -DMETIS_DIR=/Users/petra1/work/installs/metis-4.0.3 .. && make -j && make install
```

UMFPACK (part of SuiteSparse) and METIS need to be provided as shown above.

# Interfacing with HiOp

If your NLP is structured, it may be beneficial to use HiOp. If your NLP is unstructured, then you should be looking at a general purpose NLP solver such as the open-source [Ipopt](https://github.com/coin-or/Ipopt).    

HiOp supports two input formats: `hiopInterfaceDenseConstraints` and `hiopInterfaceMDS`. Both formats are in the form of C++ interfaces (e.g., abstract classes), see [hiopInterface.hpp](src/Interface/hiopInterface.hpp) file, that the user must instantiate/implement and provide to HiOp.

*`hiopInterfaceDenseConstraints` interface* supports NLPs with **billions** of variables with and without bounds but only limited number (<100) of general, equality and inequality constraints. The underlying algorithm is a limited-memory quasi-Newton interior-point method and generally scales well computationally (but it may not algorithmically) on thousands of core. This interface uses MPI for parallelization

*`hiopInterfaceMDS` interface* supports mixed dense-sparse NLPs and achive parallelization using GPUs. Limited speed-up can be obtained on multi-cores CPUs via multithreaded MKL. 

More information on the HiOp interfaces are [here](src/Interface/README.md).

# Acknowledgments

HiOp has been developed under the financial support of: 
- Department of Energy, Office of Advanced Scientific Computing Research (ASCR)
- Department of Energy, Advanced Research Projects Agency-Energy (ARPA‑E)
- Lawrence Livermore National Laboratory, through the LDRD program

# Copyright
Copyright (c) 2017, Lawrence Livermore National Security, LLC. All rights reserved. Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-742473. Written by Cosmin G. Petra, petra1@llnl.gov. 

HiOp is free software; you can modify it and/or redistribute it under the terms of the BSD 3-clause license. See [COPYRIGHT](/COPYRIGHT) and [LICENSE](/LICENSE) for complete copyright and license information.
 

