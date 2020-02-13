# Notes and findings on GPU linear solvers

Fast dual CPU Intel Xeon(R) Gold 6136 CPU @ 3.00GHz, each CPU with 12 cores (24 threads), for a total of 24 cores (48 threads). One Nvidia Quadro GV100 with double-precision performance of 7.4 Tflops, single-precision performance of 14.8 tflops, and tensor performance of 118.5 tflops. GPU has 32Gb of memory. 

## Symmetric indefinite factorizations

### Magma dsytrf

Bunch-Kauffman is offered with a CPU interface. GPU interface is not yet available. GPU Gflop/s is dissapointing, ranging from 100 to 300 Gflop/s for matrices of size ranging from 5k to 10k. This is about 1.2 to 3.6 % of GPU's peak. About 3 times faster than CPU with 24 MKL threads (on large matrices only).

Aasen communication-avoiding also disappointing: 200 Gflops to 400 Gflops.

As expected, the block non-pivoting version with GPU interface (factorizing "nice", diagonally dominant matrices) is 
extremely fast on GPU: 1.5 Tflops to 3.5Tflops **(20% to 47% of GPU's peak)** for matrices of size ranging 
from 5k to 10k. As much as 30 times faster than the two CPUs, which probably rely on the pivoting-based Bunch-Kauffman of MKL.

Results are in line with  http://www.netlib.org/utk/people/JackDongarra/PAPERS/dense-symmetric-indefinite.pdf https://www.icl.utk.edu/files/publications/2017/icl-utk-948-2017.pdf

### cuSolver cusolverDnDsytrf

Have not ran it, but expect (only) a slight increase in the performance for Bunch-Kauffman over Magma.

## Symmetric positive definite factorization

### Magma dpotrf

1.4 Tflops to 4.7 Tflops **(18.9% to 63% of GPU's peak)** for matrices of size ranging from 5k to 20k. As much as 4 times faster than the two CPUs (which generally achieve about 1 Tflops).

These results are in line with this [Magma paper](http://www.netlib.org/utk/people/JackDongarra/PAPERS/high-performance-cholesky-.pdf)

### cuSolver 
Have not ran it, but should be in line with the above performance of Magma.

## Observations

It appears that pivoting factorizations methods perform poorly even for dense matrices. 
It is dangerous to rely on the non-pivoting symmetric indefinite factorization when using interior-point optimization. 
One alternative would be to research the coupling of the non-pivoting symmetric indefinite factorization with iterative refinement and 
possibly with randomized butterfly transformations that are advocated Magma team, see 
[this paper](https://www.icl.utk.edu/files/publications/2017/icl-utk-948-2017.pdf). An alternative algorithmic approach 
would be to stick with the original plan and reduce the symmetric indefinite interior-point KKT system to 
a symmetric positive-definite linear system. While this does not necessarily solve the issue of numerical stability 
(and most likely will still require optimization-specific iterative refinement techniques), it does full exploit structure 
at the benefit of slightly smaller space and time complexities. The reduction will rely on matrix-matrix products and therefore, 
I expect will have the same or better peak performance as Magma's dpotrf. To keep in mind: 
one needs to also find a way to compute the inertia of the original indefinite system; also, the implementation of this 
alternative approach is quite low-level / elaborate.

The speedup of GPU over CPU is expected to drastically increase on Summit since the CPU cores are slower than the cores of the Xeon's used above (which also have quite generours amounts of L1, L2, and L3 caches).