//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file hiopLinSolverSparseCUSOLVER.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 *
 */

#ifndef HIOP_LINSOLVER_CUSOLVER
#define HIOP_LINSOLVER_CUSOLVER

#include "hiopLinSolver.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include <unordered_map>

#include "hiop_cusolver_defs.hpp"
#include "klu.h"
/** implements the linear solver class using nvidia_ cuSolver (GLU
 * refactorization)
 *
 * @ingroup LinearSolvers
 */

namespace hiop
{

constexpr double ZERO = 0.0;
constexpr double EPSILON = 1.0e-18;
constexpr double EPSMAC  = 1.0e-16;

// Forward declaration of inner IR class
class hiopLinSolverSymSparseCUSOLVERInnerIR;

class hiopLinSolverSymSparseCUSOLVER : public hiopLinSolverSymSparse
{
public:
  // constructor
  hiopLinSolverSymSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymSparseCUSOLVER();

  /** Triggers a refactorization of the matrix, if necessary.
   * Overload from base class.
   * In this case, KLU (SuiteSparse) is used to refactor*/
  int matrixChanged();

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved.
   * On exit is contains the solution(s).  */
  bool solve(hiopVector& x_);

  /** Multiple rhs not supported yet */
  virtual bool
  solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

private:
  //
  int m_;   // number of rows of the whole matrix
  int n_;   // number of cols of the whole matrix
  int nnz_; // number of nonzeros in the matrix

  int* kRowPtr_; // row pointer for nonzeros
  int* jCol_;    // column indexes for nonzeros
  double* kVal_; // storage for sparse matrix

  int* index_covert_CSR2Triplet_;
  int* index_covert_extra_Diag2CSR_;

  /** options **/
  // TODO: KS need options for:
  // (3) iterative refinement: fgmres, bicgstab
  // (4) if ir is fgmres, there are different gram-schmidt
  // options: MGS, CGS2, MGS-2 synch, MGS-1 synch/

  int ordering_;
  std::string fact_;
  std::string refact_;
  std::string use_ir_;

  /** needed for cuSolver **/

  cusolverStatus_t sp_status_;
  cusparseHandle_t handle_ = 0;
  cusolverSpHandle_t handle_cusolver_ = nullptr;
  cublasHandle_t handle_cublas_;

  cusparseMatDescr_t descr_A_, descr_M_;
  csrluInfoHost_t info_lu_ = nullptr;
  csrgluInfo_t info_M_ = nullptr;

  cusolverRfHandle_t handle_rf_ = nullptr;
  size_t buffer_size_;
  size_t size_M_;
  double* d_work_;
  int ite_refine_succ_ = 0;
  double r_nrminf_;

  // KLU stuff
  int klu_status_;
  klu_common Common_;
  klu_symbolic* Symbolic_ = nullptr;
  klu_numeric* Numeric_ = nullptr;
  /*pieces of M */
  int* mia_ = nullptr;
  int* mja_ = nullptr;

  /* for GPU data */
  double* da_;
  int* dia_;
  int* dja_;
  double* devx_;
  double* devr_;
  double* drhs_;

  int factorizationSetupSucc_;
  /* needed for cuSolverRf */
  int* d_P_;
  int* d_Q_; // permutation matrices
  double* d_T_;

  // iterative refinement

  hiopLinSolverSymSparseCUSOLVERInnerIR* ir_;

  /* private function: creates a cuSolver data structure from KLU data
   * structures. */

  /** called the very first time a matrix is factored. Perform KLU
   * factorization, allocate all aux variables
   *
   * @note Converts HiOp triplet matrix to CSR format.
   */
  virtual void firstCall();

  /** Function to compute nnz and set row pointers */
  void compute_nnz();
  /** Function to compute column indices and matrix values arrays */
  void set_csr_indices_values();

  int createM(const int n, 
              const int nnzL, 
              const int* Lp, 
              const int* Li, 
              const int nnzU, 
              const int* Up, 
              const int* Ui);

  template <typename T> void hiopCheckCudaError(T result, const char* const file, int const line);
  /* private functions needed for refactorization setup, no need to make them public */

  int initializeKLU();
  int initializeCusolverGLU();
  int initializeCusolverRf();

  int refactorizationSetupCusolverGLU();
  int refactorizationSetupCusolverRf();

  void IRsetup();
  friend class hiopLinSolverNonSymSparseCUSOLVER;
};


// Forward declaration of LU class
class hiopLinSolverSymSparseCUSOLVERLU;

class hiopLinSolverSymSparseCUSOLVERInnerIR
{

public:
  hiopLinSolverSymSparseCUSOLVERInnerIR();
  hiopLinSolverSymSparseCUSOLVERInnerIR(int restart, double tol, int maxit);
  ~hiopLinSolverSymSparseCUSOLVERInnerIR();
  int getFinalNumberOfIterations();
  double getFinalResidalNorm();
  double getInitialResidalNorm();
  // this is public on purpose, can be used internally or outside, to compute the residual.
  void fgmres(double* d_x, double* d_b);

private:
  // Krylov vectors
  double* d_V_;
  double* d_Z_;
  double final_residual_norm_;
  int fgmres_iters_;
  double initial_residual_norm_;
  int restart_;
  int maxit_;
  double tol_;
  std::string orth_option_;
  // the matrix in question
  cusparseSpMatDescr_t mat_A_;
  // needed for matvec
  cusparseDnVecDescr_t vec_x_ = NULL;
  cusparseDnVecDescr_t vec_Ax_ = NULL;
  int n_;
  // handles - MUST BE SET AT INIT
  cusparseHandle_t cusparse_handle_;
  cublasHandle_t cublas_handle_;
  cusolverRfHandle_t cusolverrf_handle_;

  // aux cariables, avoid multiple allocs at all costs

  // GPU:
  double* d_T_;
  int* d_P_;
  int* d_Q_;

  double* d_rvGPU_;
  double* d_Hcolumn_;
  double* d_H_col_;
  void* mv_buffer_; /* SpMV buffer */

  // CPU:
  double* h_L_;
  double* h_H_;
  double* h_rv_;
  // for givens rotations
  double* h_c_;
  double* h_s_;
  // for Hessenberg system
  double* h_rs_;
  // neded in some of the orthogonalization methods
  double* h_aux_;

  const double minusone_ = -1.0;
  const double one_ = 1.0;
  const double zero_ = 0.0;
  // private function needed for fgmres: orthogonalize i+1 vector against i vectors already orthogonal
  void GramSchmidt(int i);

  // matvec black-box: b = b - A*d_x if option is "residual" and b=A*x if option is "matvec"
  void cudaMatvec(double* d_x, double* d_b, std::string option);

  // not used (yet)

  hiopLinSolverSymSparseCUSOLVERLU* LU_data;
  friend class hiopLinSolverSymSparseCUSOLVER;
  friend class hiopLinSolverNonSymSparseCUSOLVER;
};

// class to store and operatate on LU data: will be needed in the future
class hiopLinSolverSymSparseCUSOLVERLU
{
public:
  // constructor
  hiopLinSolverSymSparseCUSOLVERLU();
  ~hiopLinSolverSymSparseCUSOLVERLU();
  void solve();
  void extractFromKLU();
  void extractFromRf();
  // to clear but not free the memory
  void intermediateCleanup();

private:
  // buffers needed for tri solves
  void* LBuffer_;
  void* UBuffer_;
  // needed for triangular solves
  cusparseMatDescr_t descrL_;
  cusparseMatDescr_t descrU_;
  csrsv2Info_t infoL_;
  csrsv2Info_t infoU_;
  cusparseSolvePolicy_t policy_;

  // matrix data of L and U factors
  double* d_Lx_;
  int* d_Lp_;
  int* d_Li_;

  double* d_Ux_;
  int* d_Up_;
  int* d_Ui_;

  // matrix CPU data - this is needed to avoid allocing over and over and over

  double* Lx_;
  int* Lp_;
  int* Li_;

  double* Ux_;
  int* Up_;
  int* Ui_;

  // permutation vectors
  int* d_P_ = NULL;
  int* d_Q_ = NULL;

  // aux vectors;
  double* d_x3_;
  double* d_xtemp_;
  // solve or not using this data
  // if manual is 0, then RfSolve() is used
  int manual_ = 1;

  int analysis_done_ = 0;
  friend class hiopLinSolverSymSparseCUSOLVER;
  friend class hiopLinSolverNonSymSparseCUSOLVER;
};
} // namespace hiop

#endif
