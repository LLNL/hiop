

# HiOp - HPC solver for optimization

## Build/install instructions
HiOp uses a CMake-based build system. There are no customization made to CMake, hence, all the standard CMake options apply.

A standard build can be done by invoking
```shell 
$> cmake ..
$> make 
$> make test
$> make install
```
in the 'build' directory. This sequence will build HiOp and install the headers and the shared library in the directory '_dist-default-build' in HiOp's root directory.

The installation can be customized using the standard CMake options. For example, one can provide a different directory to install HiOp's distribution files by using 
```sh
$> cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/hiop ..'
```
at step i.


### HiOp-specific build options
* Enable/disable MPI: *-DWITH_MPI=[ON/OFF]* (by default ON)
* Ultra safety checks: *-DEEP_CHECKING=[ON/OFF]* (by default ON) used for increased robustness and self-diagnostication. Disabling DEEP_CHECKING usually provides 30-40% execution speedup in HiOp.

These options can be specified in step i. by passing "-DFEATURE=VALUE" to cmake in step i.

### Other useful options to use with CMake
* *-DCMAKE_BUILD_TYPE=Release* will build the code with the optimization flags on
* *-DCMAKE_CXX_FLAGS="-O3"* will enable a high level of compiler code optimization

## Dependencies
HiOp requires LAPACK and BLAS. MPI is optional. All these dependencies are automatically detected by HiOp's build system.


## Acknowledgments

HiOp has been developed under the financial support of: 
- Lawrence Livermore National Laboratory, thorugh the LDRD program
- Department of Energy, Office of Advanced Scientific Computing Research

