# Change Log
All notable changes to HiOp are documented in this file.

## Version 0.6.1: HIP linear algebra workaround and update for RAJA > v0.14  (March 31, 2022)
This version/tag provides a workaround for an issue in the HIP BLAS and updates the RAJA code to better operate with the newer versions of RAJA.

## Version 0.6.0: Release of the PriDec optimization and improved GPU computations (March 31, 2022)
The salient features of v0.6.0 are 

- the release of the primal decomposition (PriDec) solver for structured two-stage problems 
- improved support for (NVIDIA) GPUs for solving sparse optimization problems via NVIDIA's cuSOLVER API and newly developed condensed optimization kernels.

Other notable capabilities include
 - improved accuracy in the computations of the search directions via Krylov-based iterative refinement
 - design of a matrix interface for sparse matrices in compressed sparse row format and (capable) CPU reference implementation

## Version 0.5.4: Elastic mode, Krylov solvers, and misc bug fixes (March 2, 2022)
New algorithmic features related to the NLP solver(s) and associated linear algebra KKT systems

 - soft feasibility restoration
 - Relaxer of equality constraints at the NLP formulation level
 - Krylov interfaces and implementation for CG and BiCGStab (ready for device computations)
 - protype of the condensed linear system and initial Krylov-based iterative refinement
 - update of the Magma solver class for the latest Magma API
 - elastic mode
 
This release also includes several bug fixes.

## Version 0.5.3: xSDK compliance (Dec 3, 2021)

 xSDK compliance 
 
## Version 0.5.2: xSDK compliance and misc bug fixes (Dec 2, 2021)

 - fixed bugs in the IPM solver: gradient scaling on CUDA, unscaled objective in the  user callbacks, lambda capture fix in axpy for ROCm
 - exported sparse config in cmake
 - added user options for the algorithm parameters in PriDec solver
 
## Version 0.5.1: Objective scaling factor fix (Oct 21, 2021)

Modified the computation of the scaling factor to use the user-specified initial point

## Version 0.5.0: MDS device computations, and porting of sparse kernels (Sep 30, 2021)

The salient features of this major release are

 - update of the interface to MAGMA and capability for running mixed dense-sparse (MDS) problems solely in the device memory space
 - added interface PARDISO linear solver
 - porting of the sparse linear algebra kernels to device via RAJA performance portability layer
 - various optimizations and bug fixes for the RAJA-based dense linear algebra kernels
 - Primal decomposition solver HiOp-PriDec available as a release candidate
