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
  virtual int matrixChanged();

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved.
   * On exit is contains the solution(s).  */
  virtual bool solve(hiopVector& x_);

  /** Multiple rhs not supported yet */
  virtual bool
  solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

protected:
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
  bool is_first_solve_;
  bool is_first_call_;
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
};

class hiopLinSolverSymSparseCUSOLVERGPU : public hiopLinSolverSymSparseCUSOLVER
{
public:  
  hiopLinSolverSymSparseCUSOLVERGPU(const int& n, const int& nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymSparseCUSOLVERGPU();

  virtual int matrixChanged();
  virtual bool solve(hiopVector& x_);

  /** Multiple rhs not supported yet */
  virtual bool
  solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

private:
  hiopVector* rhs_host_;
  hiopMatrixSparse* M_host_;
};

/**
 * @brief Iterative refinement class
 * 
 */
class hiopLinSolverSymSparseCUSOLVERInnerIR
{

public:
  hiopLinSolverSymSparseCUSOLVERInnerIR();
  hiopLinSolverSymSparseCUSOLVERInnerIR(int restart, double tol, int maxit);
  ~hiopLinSolverSymSparseCUSOLVERInnerIR();
  int setup(cusparseHandle_t cusparse_handle,
            cublasHandle_t cublas_handle,
            cusolverRfHandle_t cusolverrf_handle,
            int n,
            double* d_T,
            int* d_P,
            int* d_Q,
            double* devx,
            double* devr);

  int getFinalNumberOfIterations();
  double getFinalResidalNorm();
  double getInitialResidalNorm();
  // this is public on purpose, can be used internally or outside, to compute the residual.
  void fgmres(double* d_x, double* d_b);
  void set_tol(double tol) {tol_ = tol;} ///< Set tolerance for the Krylov solver

  /**
   * @brief Set the up system matrix object mat_A_ of type cusparseSpMatDescr_t
   * 
   * @param n    - size of the matrix
   * @param nnz  - number of nonzeros in the matrix
   * @param irow - array of row pointers
   * @param jcol - array of column indices
   * @param val  - array of sparse matrix values
   * 
   * @return int
   * 
   * @pre Arrays `irow`, `jcol` and `val` are on the device.
   */
  int setup_system_matrix(int n, int nnz, int* irow, int* jcol, double* val);

  // Simple accessors
  int& maxit()
  {
    return maxit_;
  }

  double& tol()
  {
    return tol_;
  }

  std::string& orth_option()
  {
    return orth_option_;
  }

  int& restart()
  {
    return restart_;
  }

  // cusparseSpMatDescr_t& mat_A()
  // {
  //   return mat_A_;
  // }

  int& conv_cond()
  {
    return conv_cond_;
  }

private:
  // Krylov vectors
  double* d_V_{ nullptr };
  double* d_Z_{ nullptr };
  double final_residual_norm_;
  int fgmres_iters_;
  double initial_residual_norm_;
  int restart_;
  int maxit_;
  double tol_;
  int conv_cond_; // convergence condition, can be 0, 1, 2 for IR
  std::string orth_option_;
  // the matrix in question
  cusparseSpMatDescr_t mat_A_;
  int* dia_{ nullptr };
  int* dja_{ nullptr };
  double* da_{ nullptr };
  // needed for matvec
  cusparseDnVecDescr_t vec_x_{ nullptr };
  cusparseDnVecDescr_t vec_Ax_{ nullptr };
  int n_;
  int nnz_;
  // handles - MUST BE SET AT INIT
  cusparseHandle_t cusparse_handle_{ nullptr };
  cublasHandle_t cublas_handle_{ nullptr };
  cusolverRfHandle_t cusolverrf_handle_{ nullptr };
  cusolverSpHandle_t cusolver_handle_{ nullptr };
  // aux cariables, avoid multiple allocs at all costs

  // GPU:
  double* d_T_{ nullptr };
  int* d_P_{ nullptr };
  int* d_Q_{ nullptr };

  double* d_rvGPU_{ nullptr };
  double* d_Hcolumn_{ nullptr };
  double* d_H_col_{ nullptr };
  void* mv_buffer_{ nullptr }; /* SpMV buffer */

  // CPU:
  double* h_L_{ nullptr };
  double* h_H_{ nullptr };
  double* h_rv_{ nullptr };
  // for givens rotations
  double* h_c_{ nullptr };
  double* h_s_{ nullptr };
  // for Hessenberg system
  double* h_rs_{ nullptr };
  // neded in some of the orthogonalization methods
  double* h_aux_{ nullptr };

  const double minusone_ = -1.0;
  const double one_ = 1.0;
  const double zero_ = 0.0;
  // private function needed for fgmres: orthogonalize i+1 vector against i vectors already orthogonal
  void GramSchmidt(int i);

  // matvec black-box: b = b - A*d_x if option is "residual" and b=A*x if option is "matvec"
  void cudaMatvec(double* d_x, double* d_b, std::string option);
  //KS: needed for testing -- condider delating later
  double matrixAInfNrm();
  double vectorInfNrm(int n, double* d_v);
  //end of testing
  

  template <typename T>
  void hiopCheckCudaError(T result, const char* const file, int const line);

};

/**
 * @brief Class to store and operatate on LU data: will be needed in the future
 * 
 */
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

};


} // namespace hiop

#endif
