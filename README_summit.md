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
In some case, `cmake` picks up a different gcc compiler than the one loaded via `module`. This situation can be remedied by preloading the `CC` and `CXX` for `cmake` as it is done above.