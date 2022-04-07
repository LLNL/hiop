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
    virtual bool solve(hiopMatrix& /* x */)
    {
      assert(false && "not yet supported");
      return false;
    }

    int refactorizationSetupCusolverGLU();
    int refactorizationSetupCusolverRf();

    int initializeKLU();
    int initializeCusolverGLU();
    int initializeCusolverRf();

    void setFactorizationType(std::string newFact_);
    void setRefactorizationType(std::string newRefact_);

    std::string getFactorizationType();
    std::string getRefactorizationType();
    // KS: can consider one global function to set all the options but thats
    // TBA in the future.    

    void setIRParameters(int restart, int maxit, double tol,  std::string gsType);
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
    //(1) factorization (cusolver-glu, cusolver-rf, klu)
    //(2) refactorization: if factorization is klu, it can be followed either
    // by
    // cusolver-glu or cusolver-rf factorization (3) iterative refinement:
    // fgmres, bicgstab (4) if ir is fgmres, there are different gram-schmidt
    // options: MGS, CGS2, MGS-2 synch, MGS-1 synch/

    std::string fact_;
    std::string refact_;

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
    int* d_P;
    int* d_Q; // permutation matrices
    double* d_T;
    /* private function: creates a cuSolver data structure from KLU data
     * structures. */

    int createM(const int n, 
                const int nnzL, 
                const int* Lp, 
                const int* Li, 
                const int nnzU, 
                const int* Up, 
                const int* Ui);

    template <typename T>
    void hiopCheckCudaError(T result, const char* const file, int const line);


    /** KS: needed for FGMRES_IR */
    int useIR_;
    double* d_V_;
    double* d_Z_;
    double* d_LH_;
    double* d_w_;
    double* computeResidual(double* x);
    /** KS: experimental feature; needed if cuSolverRfSolve() fails with code 6*/
    void boostAndRefactorize();
    /** for Gram-Schmidt */
    void GramSchmidt();
    /** actual IR */
    void fgmres(double* b, double* x);
    int restart_;
    int maxIt_;
    double tol_;
    std::string gsType_; 

  public:
    /** called the very first time a matrix is factored. Perform KLU
     * factorization, allocate all aux variables */
    virtual void firstCall();

    friend class hiopLinSolverNonSymSparseCUSOLVER;
  };

  class hiopLinSolverNonSymSparseCUSOLVER : public hiopLinSolverNonSymSparse
  {
  public:
    hiopLinSolverNonSymSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp);

    virtual ~hiopLinSolverNonSymSparseCUSOLVER();

    /** Triggers a SuiteSparse KLU refactorization of the matrix, if necessary.
     * Overload from base class. */
    int matrixChanged();

    /** solves a linear system.
     * param 'x' is on entry the right hand side(s) of the system to be solved.
     * On exit is contains the solution(s).  */
    bool solve(hiopVector& x);

    /** Multiple rhs not supported yet */
    virtual bool solve(hiopMatrix& /* x */)
    {
      assert(false && "not yet supported");
      return false;
    }

    int refactorizationSetupCusolverGLU();
    int refactorizationSetupCusolverRf();

    int initializeKLU();
    int initializeCusolverGLU();
    int initializeCusolverRf();

    void setFactorizationType(std::string newFact_);
    void setRefactorizationType(std::string newRefact_);

    std::string getFactorizationType();
    std::string getRefactorizationType();

  private:
    int m_;   // number of rows of the whole matrix
    int n_;   // number of cols of the whole matrix
    int nnz_; // number of nonzeros in the matrix

    int* kRowPtr_; // row pointer for nonzeros
    int* jCol_;    // column indexes for nonzeros
    double* kVal_; // storage for sparse matrix

    int* index_covert_CSR2Triplet_;
    int* index_covert_extra_Diag2CSR_;
    std::unordered_map<int, int> extra_dia_g_nnz_map;

    std::string fact_;
    std::string refact_;
    int factorizationSetupSucc_;
    /** needed for CUSOLVER and KLU */

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
    int Atype;
    /*pieces of M */
    int* mia_ = nullptr;
    int* mja_ = nullptr;
    /*GPU vatiables */
    double* da_;
    int* dia_;
    int* dja_;
    double* devx_;
    double* devr_;
    double* drhs_;
    /* needed for cuSolverRf */
    int* d_P;
    int* d_Q; // permutation matrices
    double* d_T;
    /* private function: creates a cuSolver data structure from KLU data
     * structures. */

    int createM(const int n, 
                const int nnzL, 
                const int* Lp, 
                const int* Li,
                const int nnzU, 
                const int* Up, 
                const int* Ui);

    template <typename T> void hiopCheckCudaError(T result, const char* const file, int const line);

  public:
    /** called the very first time a matrix is factored. */
    void firstCall();

    friend class hiopLinSolverSymSparseCUSOLVER;
  };

} // namespace hiop

#endif
