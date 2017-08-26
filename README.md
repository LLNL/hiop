

# HiOp - HPC solver for optimization

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
* Enable/disable MPI: *-DWITH_MPI=[ON/OFF]* (by default ON)
* Ultra safety checks: *-DEEP_CHECKING=[ON/OFF]* (by default ON) used for increased robustness and self-diagnostication. Disabling DEEP_CHECKING usually provides 30-40% execution speedup in HiOp.

For example:
```shell 
$> cmake -DWITH_MPI=ON -DEEP_CHECKING=ON ..
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
- Lawrence Livermore National Laboratory, thorugh the LDRD program
- Department of Energy, Office of Advanced Scientific Computing Research

