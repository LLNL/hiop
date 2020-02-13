# Super plan

Developement action items needed to enable solving ACOPF on GPUs (Phase1) and then SC-ACOPF (Phase2). 

## Phase 1 - ACOPF with dense linear algebra via the mixed dense-sparse (MDS) formulation and linear algebra

An MDS nonlinear programming (NLP) formulation means Jacobian is split in a sparse left block and dense right block (of columns/variables). The MDS Hessian is expected as a diagonal sparse block and a diagonal dense block (in other words the objective and constraints are separable in the sparse and dense variables).

## A. Mixed Dense-Sparse Interface

1. Design of the HiOp (abstract) C++ interface to support problems with mixed dense-sparse derivatives (Jacobian and Hessian)
2. Implementation of linear algebra objects to support the above
3. Refactoring of HiOp's internals (internal problem formulation and evaluation and IPM solver class)
4. Implementation of a simple, standalone MDS example
5. Update of HiOp's adapter to Ipopt's TNLP interface (to facilitate testing)

Code [HiOp branch](https://github.com/LLNL/hiop/tree/dev/block_interface) -> [PR#11](https://github.com/LLNL/hiop/pull/11)

*Completed and merged in the master*
 
## B. Revisit the IPM filter line-search algorithm

1. Refactor existing quasi-Newton IPM filter line-search solver class to share baseline code with the new full-Hessians IPM solver class
2. Implement (a minimal set of) additional algorithm features needed by full Newton IPM line-search (will follow [Ipopt](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf))
3. Implement initial (dense) KKT full-Newton linear solver class (for testing the IPM Newton solver)
4. Revisit 2. and implement MDS KKT full-Newton linear solver class and the required linear algebra objects (diagonally scaled sparse mat-mattrans mults, sparse mat-dense mat mults, etc)

*Completed.* [HiOp branch dev/NewtonMDS](https://github.com/LLNL/hiop/tree/dev/NewtonMDS)

## C. GPU linear algebra

1. Porting B.4. to GPUs via an existing off-the-shelf linear solver
2. Profiling on Summit with the HiOp's simple MDS Ex4 problem.

Driver for hybrid GPU-CPU driver: HiOp's Drivers/nlpMDS_ex4.exe

*Completed.* [HiOp branch dev/NewtonMDS](https://github.com/LLNL/hiop/tree/dev/NewtonMDS)

C.2: preliminary results: HiOp + magma_dsysv_nopiv_gpu capable of up to 4.1 TFlops and 2.9 TFlops counting CPU-to-GPU-to-CPU data transfer time. Tests done on a Nvidia Quadro GV100 with double-precision performance of 7.4 Tflops. Evaluation on Summit will be done at a later time.

## D. Kron reduction of sparse ACOPF to MDS ACOPF

*In progress*

1. Interface in-out design
2. Implementation of the reduction
3. Instantiation of the interface for in=gollnlp and out=hiop_mds
4. Revisit B.2. - likely additional algorithmic features will be needed to ensure robustness of solver

## E. Summit runs and profiling of ACOPF

## F. Final adjustments
Expected to need adjustments of A.1. to support advanced primal-dual restarts for binding lines