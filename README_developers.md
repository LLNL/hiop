# Minimal testing procedure 
Any PR or push to the master should go through and pass the procedures below. The shell commands need to be ran in the 'build' directory and assume bash shell. 

## 1. All tests in 'make test' pass for
1. a. MPI=ON, DEEPCHECKS=OFF, RELEASE=ON (this is the high-performance version of HiOp)
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
1. b. MPI=OFF, DEEPCHECKS=OFF, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
and, optionally, for 
1. c. MPI=ON, DEEPCHECKS=ON, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
1. d. MPI=OFF, DEEPCHECKS=ON, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```

1. e. MPI=ON, DEEPCHECKS=OFF, DEBUG=ON 
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4; make install; make test
```

1. f. MPI=OFF, DEEPCHECKS=OFF, DEBUG=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4; make install; make test
```

## 2. Valgrind reports no errors and no warning when running the testing drivers of HiOp. Mandatory on Linux, optional on MacOS
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex1.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex2.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex3.exe 
```

## 3. clang with fsanitize group checks reports no warning and no errors. MacOS only.
```shell
$> rm -rf *; CC=clang CXX=clang++ cmake -DCMAKE_CXX_FLAGS="-fsanitize=nullability,undefined,integer,alignment" -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex1.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex2.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex3.exe 
```

