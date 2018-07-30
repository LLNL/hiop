

# HiOp - HPC solver for optimization
HiOp is an optimization solver for solving certain mathematical optimization problems expressed as nonlinear programming problems. HiOp is a lightweight HPC solver that leverages application's existing data parallelism to parallelize the optimization iterations by using specialized linear algebra kernels.

## Build/install instructions
HiOp uses a CMake-based build system. There are no customization made to CMake, hence, all the standard CMake options apply.

A standard build can be done by invoking in the 'build' directory the following 
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

## Dependencies
HiOp requires LAPACK and BLAS. MPI is optional. All these dependencies are automatically detected by the build system.


## Acknowledgments

HiOp has been developed under the financial support of: 
- Lawrence Livermore National Laboratory, through the LDRD program
- Department of Energy, Office of Advanced Scientific Computing Research

## Copyright
Copyright (c) 2017, Lawrence Livermore National Security, LLC. All rights reserved. Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-742473. Written by Cosmin G. Petra, petra1@llnl.gov. 

HiOp is free software; you can modify it and/or redistribute it under the terms of the BSD 3-clause license. See [COPYRIGHT](/COPYRIGHT) and [LICENSE](/LICENSE) for complete copyright and license information.
 

