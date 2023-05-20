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
 * @file IterativeRefinement.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#pragma once

#include "klu.h"
#include "hiop_cusolver_defs.hpp"
#include <string>

namespace hiop{
  class hiopLinSolverSymSparseCUSOLVER;
}

namespace ReSolve {

constexpr double ZERO = 0.0;
constexpr double EPSILON = 1.0e-18;
constexpr double EPSMAC  = 1.0e-16;

/**
 * @brief Iterative refinement class
 * 
 */
class IterativeRefinement
{

public:
  IterativeRefinement();
  IterativeRefinement(int restart, double tol, int maxit);
  ~IterativeRefinement();
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

  int& conv_cond()
  {
    return conv_cond_;
  }

private:
  // Krylov vectors
  double* d_V_{ nullptr };
  double* d_Z_{ nullptr };

  double final_residual_norm_;
  double initial_residual_norm_;
  int fgmres_iters_;

  // Solver parameters
  int restart_;
  int maxit_;
  double tol_;
  int conv_cond_; ///< convergence condition, can be 0, 1, 2 for IR
  std::string orth_option_;

  // System matrix data
  int n_;
  int nnz_;
  int* dia_{ nullptr };
  int* dja_{ nullptr };
  double* da_{ nullptr };
  cusparseSpMatDescr_t mat_A_;

  // Matrix-vector product data
  cusparseDnVecDescr_t vec_x_{ nullptr };
  cusparseDnVecDescr_t vec_Ax_{ nullptr };

  // CUDA libraries handles - MUST BE SET AT INIT
  cusparseHandle_t cusparse_handle_{ nullptr };
  cublasHandle_t cublas_handle_{ nullptr };
  cusolverRfHandle_t cusolverrf_handle_{ nullptr };
  cusolverSpHandle_t cusolver_handle_{ nullptr };

  // GPU data (?)
  double* d_T_{ nullptr };
  int* d_P_{ nullptr };
  int* d_Q_{ nullptr };

  double* d_rvGPU_{ nullptr };
  double* d_Hcolumn_{ nullptr };
  double* d_H_col_{ nullptr };
  void* mv_buffer_{ nullptr }; ///< SpMV buffer

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

  /**
   * @brief orthogonalize i+1 vector against i vectors already orthogonal
   * 
   * Private function needed for FGMRES.
   * 
   * @param[in] i - number of orthogonal vectors
   */
  void GramSchmidt(int i);

  /**
   * @brief matvec black-box: b = b - A*d_x if option is "residual" and b=A*x 
   * if option is "matvec"
   * 
   * @param d_x 
   * @param d_b 
   * @param option
   * 
   * @todo Document d_x and d_b; are both of them modified in this function?
   */
  void cudaMatvec(double* d_x, double* d_b, std::string option);
 
  //KS: needed for testing -- condider delating later
  double matrixAInfNrm();
  double vectorInfNrm(int n, double* d_v);
  //end of testing
  
  /**
   * @brief Check for CUDA errors.
   * 
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template <typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);

};



class MatrixCsr
{
public:
  MatrixCsr();
  ~MatrixCsr();
  void allocate_size(int n);
  void allocate_nnz(int nnz);

  int* get_irows()
  {
    return irows_;
  }

  const int* get_irows() const
  {
    return irows_;
  }

  int* get_jcols()
  {
    return jcols_;
  }

  double* get_vals()
  {
    return vals_;
  }

  int* get_irows_host()
  {
    return irows_host_;
  }

  int* get_jcols_host()
  {
    return jcols_host_;
  }

  double* get_vals_host()
  {
    return vals_host_;
  }

  void update_from_host_mirror();
  void copy_to_host_mirror();

private:
  int n_{ 0 };
  int nnz_{ 0 };

  int* irows_{ nullptr };
  int* jcols_{ nullptr };
  double* vals_{ nullptr};

  int* irows_host_{ nullptr };
  int* jcols_host_{ nullptr };
  double* vals_host_{ nullptr};


  /**
   * @brief Check for CUDA errors.
   * 
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template <typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);

};  

/**
 * @brief Class to store and operatate on LU data: will be needed in the future
 * 
 */
class RefactorizationSolver
{
public:
  // constructor
  RefactorizationSolver();
  RefactorizationSolver(int n);
  ~RefactorizationSolver();

  void enable_iterative_refinement();
  void setup_iterative_refinement_matrix(int n, int nnz);
  void configure_iterative_refinement(cusparseHandle_t   cusparse_handle,
                                      cublasHandle_t     cublas_handle,
                                      cusolverRfHandle_t cusolverrf_handle,
                                      int n,
                                      double* d_T,
                                      int* d_P,
                                      int* d_Q,
                                      double* devx,
                                      double* devr);

  /**
   * @brief Set the number of nonzeros in system matrix.
   * 
   * @param nnz 
   */
  void set_nnz(int nnz)
  {
    nnz_ = nnz;
  }

  /**
   * @brief Set up factorization of the first linear system.
   * 
   * @return int 
   */
  int setup_factorization();

  /**
   * @brief Factorize system matrix
   * 
   * @return int - factorization status: success=0, failure=-1
   */
  int factorize();

  /**
   * @brief Set the up the refactorization
   * 
   */
  void setup_refactorization();

  /**
   * @brief Refactorize system matrix
   * 
   * @return int 
   */
  int refactorize();

  /**
   * @brief Invokes triangular solver given matrix factors
   * 
   * @param dx 
   * @param tol 
   * @return bool 
   */
  bool triangular_solve(double* dx, const double* rhs, double tol);


private:
  int n_{ 0 };   ///< Size of the linear system
  int nnz_{ 0 }; ///< Number of nonzeros in the system's matrix

  MatrixCsr* mat_A_csr_{ nullptr };    ///< System matrix in nonsymmetric CSR format
  IterativeRefinement* ir_{ nullptr }; ///< Iterative refinement class

  bool cusolver_glu_enabled_{ false };         ///< cusolverGLU on/off flag
  bool cusolver_rf_enabled_{ false };          ///< cusolverRf on/off flag
  bool iterative_refinement_enabled_{ false }; ///< Iterative refinement on/off flag
  bool is_first_solve_{ true };                ///< If it is first call to triangular solver

  // Options
  int ordering_{ -1 };
  std::string fact_;
  std::string refact_;
  std::string use_ir_;

  /** needed for cuSolver **/

  cusolverStatus_t sp_status_;
  cusparseHandle_t handle_ = 0;
  cusolverSpHandle_t handle_cusolver_ = nullptr;
  cublasHandle_t handle_cublas_;

  cusparseMatDescr_t descr_A_;
  cusparseMatDescr_t descr_M_;
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
  double* devx_;
  double* devr_;

  /* needed for cuSolverRf */
  int* d_P_;
  int* d_Q_; // permutation matrices
  double* d_T_;

  /**
   * @brief Function that computes M = (L-I) + U
   * 
   * @param n 
   * @param nnzL 
   * @param Lp 
   * @param Li 
   * @param nnzU 
   * @param Up 
   * @param Ui 
   * @return int 
   */
  int createM(const int n, 
              const int nnzL, 
              const int* Lp, 
              const int* Li, 
              const int nnzU, 
              const int* Up, 
              const int* Ui);

  int initializeKLU();
  int initializeCusolverGLU();
  int initializeCusolverRf();

  int refactorizationSetupCusolverGLU();
  int refactorizationSetupCusolverRf();


  /**
   * @brief Check for CUDA errors.
   * 
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template <typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);

  friend class hiop::hiopLinSolverSymSparseCUSOLVER;
};


} // namespace ReSolve
