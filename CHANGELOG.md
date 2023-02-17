# Change Log
All notable changes to HiOp are documented in this file.

## Version 0.7.2: Execution spaces abstractions and misc fixes
This release hosts a series of comprehensive internal developments and software re-engineering to improve the portability and performance on accelerators/GPU platforms.

### Notable new features

A new "execution space" abstraction is introduced to allow multiple "backends" to run concurrently, e.g., Cuda/Hip linear algebra and Raja linear algebra. The proposed design differentiates between memory backend and execution policies to allow using Raja with Umpire-managed memory, Raja with Cuda- or Hip-managed memory, Raja with std memory. 

* Execution spaces: support for memory backends and execution policies by @cnpetra in https://github.com/LLNL/hiop/pull/543
* Build: Cuda without raja  by @cnpetra in https://github.com/LLNL/hiop/pull/579
* Update of RAJA-based dense matrix to support runtime execution spaces by @cnpetra in https://github.com/LLNL/hiop/pull/580
* Reorganization of device namespace  by @cnpetra in https://github.com/LLNL/hiop/pull/582
* RAJA Vector int with ExecSpace by @cnpetra in https://github.com/LLNL/hiop/pull/583
* Instrumentation of host vectors with execution spaces by @cnpetra in https://github.com/LLNL/hiop/pull/584
* Remove copy from/to device methods in vector classes by @cnpetra in https://github.com/LLNL/hiop/pull/587
 
New vector classes using vendor-provided API were introduced and documentation was updated/improved
* Development of `hiopVectorCuda` by @nychiang in https://github.com/LLNL/hiop/pull/572
* Implementation of `hiopVectorHip` by @nychiang in https://github.com/LLNL/hiop/pull/590
* Update user manual by @nychiang in https://github.com/LLNL/hiop/pull/591
* Update the code comments in `hiopVector` classes by @nychiang in https://github.com/LLNL/hiop/pull/592
* 
### Bug fixes
* Add support of Raja with OpenMP into Quartz CI by @nychiang in https://github.com/LLNL/hiop/pull/566
* Refine the computation in normal equation system by @nychiang in https://github.com/LLNL/hiop/pull/530
* Fix static culibos issue #567 by @nychiang in https://github.com/LLNL/hiop/pull/568
* Fix segfault, remove nonsymmetric ginkgo solver by @fritzgoebel in https://github.com/LLNL/hiop/pull/548
* Calculate the inverse objective scale correctly. by @tepperly in https://github.com/LLNL/hiop/pull/570
* Fix `hiopVectorRajaPar::copyToStartingAt_w_pattern` by @nychiang in https://github.com/LLNL/hiop/pull/569
* Gitlab pipeline refactor by @CameronRutherford in https://github.com/LLNL/hiop/pull/597

### New Contributors
* @tepperly made their first contribution in https://github.com/LLNL/hiop/pull/570

**Full Changelog**: https://github.com/LLNL/hiop/compare/v0.7.1...v0.7.2

## Version 0.7.1: Miscellaneous fixes to build system
This minor release fixes a couple of issues found in the build system after the major release 0.7 of HiOp. 

## Version 0.7.0: Fortran interface and misc fixes and improvements:
- Fortran interface and examples
- Bug fixing for sparse device linear solvers
- Implementation of CUDA CSR matrices
- Iterative refinement within CUSOLVER linear solver class
- Improved robustness and performance of mixed dense-sparse solver for AMD/HIP

## Version 0.6.2:  Initial ginkgo solver integration and misc fixes
This tag provides an initial integration with ginko, fixes a couple of issues, and add options for (outer) iterative refinement.

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
