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
 * @file hiopLinSolverSparseCUSOLVER.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 *
 */

#include "hiopLinSolverSparseCUSOLVER.hpp"

#include "hiop_blasdefs.hpp"

#include "cusparse_v2.h"
#include "klu.h"
#include <sstream>
#include <string>

#define checkCudaErrors(val) hiopCheckCudaError((val), __FILE__, __LINE__)

namespace hiop
{
  hiopLinSolverSymSparseCUSOLVER::hiopLinSolverSymSparseCUSOLVER(const int& n,
                                                                 const int& nnz,
                                                                 hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp), 
      kRowPtr_{ nullptr },
      jCol_{ nullptr }, 
      kVal_{ nullptr },
      index_covert_CSR2Triplet_{ nullptr },
      index_covert_extra_Diag2CSR_{ nullptr }, 
      n_{ n }, 
      nnz_{ 0 },
      ordering_{ 1 },
      fact_{ "klu" },   // default
      refact_{ "glu" }, // default
      factorizationSetupSucc_{ 0 }
  {
    // handles
    cusparseCreate(&handle_);
    cusolverSpCreate(&handle_cusolver_);
    cublasCreate(&handle_cublas_);

    // descriptors
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    // Set user selected options
    std::string ordering = nlp_->options->GetString("linear_solver_sparse_ordering");
    if(ordering == "amd_ssparse") {
      ordering_ = 0;
    } else if(ordering == "colamd_ssparse") {
      ordering_ = 1;
    } else {
      nlp_->log->printf(hovWarning, 
                        "Ordering %s not compatible with cuSOLVER LU, using default ...\n",
                        ordering.c_str());
      ordering_ = 1;
    }

    fact_ = nlp_->options->GetString("cusolver_lu_factorization");
    if(fact_ != "klu") {
      nlp_->log->printf(hovWarning,
                        "Factorization %s not compatible with cuSOLVER LU, using default ...\n",
                        fact_.c_str());
      fact_ = "klu";
    }

    refact_ = nlp_->options->GetString("cusolver_lu_refactorization");
    if(refact_ != "glu" && refact_ != "rf") {
      nlp_->log->printf(hovWarning, 
                        "Refactorization %s not compatible with cuSOLVER LU, using default ...\n",
                        refact_.c_str());
      refact_ = "glu";
    }
    // std::cout << "Ordering: " << ordering_ << ", Fact: " << fact_ << ", Refatc: " << refact_ << "\n";
  }

  hiopLinSolverSymSparseCUSOLVER::~hiopLinSolverSymSparseCUSOLVER()
  {
    // Delete CSR matrix on CPU
    delete[] kRowPtr_;
    delete[] jCol_;
    delete[] kVal_;

    // Delete CSR <--> triplet mappings
    delete[] index_covert_CSR2Triplet_;
    delete[] index_covert_extra_Diag2CSR_;

    // Delete CSR matrix on GPU
    cudaFree(dia_);
    cudaFree(da_);
    cudaFree(dja_);

    // Delete residual and solution vectors
    cudaFree(devr_);
    cudaFree(devx_);

    // Delete workspaces and handles
    cudaFree(d_work_);
    cusparseDestroy(handle_);
    cusolverSpDestroy(handle_cusolver_);
    cublasDestroy(handle_cublas_);
    cusparseDestroyMatDescr(descr_A_);

    // Delete matrix descriptor used in cuSolverGLU setup
    if(refact_ == "glu") {
      cusparseDestroyMatDescr(descr_M_);
      cusolverSpDestroyGluInfo(info_M_);
    }

    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
    delete [] mia_;
    delete [] mja_;
  }
  
  int hiopLinSolverSymSparseCUSOLVER::matrixChanged()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if(!kRowPtr_) {
      this->firstCall();
    } else {
      // update matrix
      for(int k = 0; k < nnz_; k++) {
        kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i = 0; i < n_; i++) {
        if(index_covert_extra_Diag2CSR_[i] != -1)
          kVal_[index_covert_extra_Diag2CSR_[i]]
              += M_->M()[M_->numberOfNonzeros() - n_ + i];
      }
      // somehow update the matrix not sure how
    } // else

    if((Numeric_ == nullptr) && (factorizationSetupSucc_ == 0)) {
      Numeric_ = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic_, &Common_);
      if(Numeric_ == nullptr) {
        nlp_->log->printf(hovWarning, "Numeric klu factorization failed. Regularizing ...\n");
        // This is not a catastrophic failure
        // The matrix is singular so return -1 to regularaize!
        return -1;
      } else { // Numeric was succesfull so now can set up
        factorizationSetupSucc_ = 1;
        nlp_->log->printf(hovScalars, "Numeric klu factorization succesful! \n");
        if(refact_ == "glu") {
          this->initializeCusolverGLU();
          this->refactorizationSetupCusolverGLU();
        } else if(refact_ == "rf") {
          this->initializeCusolverRf();
          this->refactorizationSetupCusolverRf();
        } else { // for future -
          assert(0 && "Only glu and rf refactorizations available.\n");
        }
      }
    } else { // Numeric_ != nullptr
      checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
      // re-factor here
      if(refact_ == "glu") {
        sp_status_ = cusolverSpDgluReset(handle_cusolver_, 
                                         n_,
                                         /* A is original matrix */
                                         nnz_,
                                         descr_A_,
                                         da_,
                                         dia_,
                                         dja_,
                                         info_M_);

        sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
      } else {
        if(refact_ == "rf") {
          sp_status_ = cusolverRfResetValues(n_, 
                                             nnz_, 
                                             dia_,
                                             dja_,
                                             da_,
                                             d_P,
                                             d_Q,
                                             handle_rf_);
          cudaDeviceSynchronize();
          sp_status_ = cusolverRfRefactor(handle_rf_);
        }
      }
      // end of factor
    }
    return 0;
  }

  bool hiopLinSolverSymSparseCUSOLVER::solve(hiopVector& x)
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);
    assert(x.get_size() == M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVector* rhs = x.new_copy();
    double* dx = x.local_data();
    double* drhs = rhs->local_data();
    checkCudaErrors(cudaMemcpy(devr_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    // solve HERE
    if(refact_ == "glu") {
      sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                       n_,
                                       /* A is original matrix */
                                       nnz_,
                                       descr_A_,
                                       da_,
                                       dia_,
                                       dja_,
                                       devr_,/* right hand side */
                                       devx_,/* left hand side */
                                       &ite_refine_succ_,
                                       &r_nrminf_,
                                       info_M_,
                                       d_work_);
      if(sp_status_ == 0) {
        checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
      } else {
        nlp_->log->printf(hovError,  // catastrophic failure
                          "Solve failed with starus: %d\n", 
                          sp_status_);
        return false;
      }
    } else {
      if(refact_ == "rf") {
        if(Numeric_ == nullptr) {
          sp_status_ = cusolverRfSolve(handle_rf_,
                                       d_P,
                                       d_Q,
                                       1,
                                       d_T,
                                       n_,
                                       devr_,
                                       n_);
          if(sp_status_ == 0) {
            checkCudaErrors(cudaMemcpy(dx, devr_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
          } else {
            nlp_->log->printf(hovError,  // catastrophic failure
                              "Solve failed with starus: %d\n", 
                              sp_status_);
            return false;
          }
        } else {
          memcpy(dx, drhs, sizeof(double) * n_);
          int ok = klu_solve(Symbolic_, Numeric_, n_, 1, dx, &Common_);
          klu_free_numeric(&Numeric_, &Common_);
          klu_free_symbolic(&Symbolic_, &Common_);
        }
      } else {
        nlp_->log->printf(hovError, // catastrophic failure
                          "Unknown refactorization, exiting\n");
        assert(false && "Only GLU and cuSolverRf are available refactorizations.");
      }
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    delete rhs;
    rhs = nullptr;
    return 1;
  }

  void hiopLinSolverSymSparseCUSOLVER::firstCall()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    kRowPtr_ = new int[n_ + 1]{ 0 };
    //
    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the
    // sparse matrix, and the 2nd part are the additional dia_gonal elememts
    // the 1st part is sorted by row
    {
      //
      // compute nnz in each row
      //
      // off-dia_gonal part
      kRowPtr_[0] = 0;
      for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
        if(M_->i_row()[k] != M_->j_col()[k]) {
          kRowPtr_[M_->i_row()[k] + 1]++;
          kRowPtr_[M_->j_col()[k] + 1]++;
          nnz_ += 2;
        }
      }
      // dia_gonal part
      for(int i = 0; i < n_; i++) {
        kRowPtr_[i + 1]++;
        nnz_ += 1;
      }
      // get correct row ptr index
      for(int i = 1; i < n_ + 1; i++) {
        kRowPtr_[i] += kRowPtr_[i - 1];
      }
      assert(nnz_ == kRowPtr_[n_]);

      kVal_ = new double[nnz_]{ 0.0 };
      jCol_ = new int[nnz_]{ 0 };
    }
    {
      //
      // set correct col index and value
      //
      index_covert_CSR2Triplet_ = new int[nnz_];
      index_covert_extra_Diag2CSR_ = new int[n_];

      int* nnz_each_row_tmp = new int[n_]{ 0 };
      int total_nnz_tmp{ 0 }, nnz_tmp{ 0 }, rowID_tmp, colID_tmp;
      for(int k = 0; k < n_; k++) {
        index_covert_extra_Diag2CSR_[k] = -1;
      }

      for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
        rowID_tmp = M_->i_row()[k];
        colID_tmp = M_->j_col()[k];
        if(rowID_tmp == colID_tmp) {
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          kVal_[nnz_tmp] += M_->M()[M_->numberOfNonzeros() - n_ + rowID_tmp];
          index_covert_extra_Diag2CSR_[rowID_tmp] = nnz_tmp;

          nnz_each_row_tmp[rowID_tmp]++;
          total_nnz_tmp++;
        } else {
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_tmp = nnz_each_row_tmp[colID_tmp] + kRowPtr_[colID_tmp];
          jCol_[nnz_tmp] = rowID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_each_row_tmp[rowID_tmp]++;
          nnz_each_row_tmp[colID_tmp]++;
          total_nnz_tmp += 2;
        }
      }
      // correct the missing dia_gonal term
      for(int i = 0; i < n_; i++) {
        if(nnz_each_row_tmp[i] != kRowPtr_[i + 1] - kRowPtr_[i]) {
          assert(nnz_each_row_tmp[i] == kRowPtr_[i + 1] - kRowPtr_[i] - 1);
          nnz_tmp = nnz_each_row_tmp[i] + kRowPtr_[i];
          jCol_[nnz_tmp] = i;
          kVal_[nnz_tmp] = M_->M()[M_->numberOfNonzeros() - n_ + i];
          index_covert_CSR2Triplet_[nnz_tmp] = M_->numberOfNonzeros() - n_ + i;
          total_nnz_tmp += 1;

          std::vector<int> ind_temp(kRowPtr_[i + 1] - kRowPtr_[i]);
          std::iota(ind_temp.begin(), ind_temp.end(), 0);
          std::sort(ind_temp.begin(), ind_temp.end(), 
            [&](int a, int b) {
              return jCol_[a + kRowPtr_[i]] < jCol_[b + kRowPtr_[i]];
            }
          );

          reorder(kVal_ + kRowPtr_[i], ind_temp, kRowPtr_[i + 1] - kRowPtr_[i]);
          reorder(index_covert_CSR2Triplet_ + kRowPtr_[i], ind_temp, kRowPtr_[i + 1] - kRowPtr_[i]);
          std::sort(jCol_ + kRowPtr_[i], jCol_ + kRowPtr_[i + 1]);
        }
      }
      delete[] nnz_each_row_tmp;
    }

    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));

    checkCudaErrors(cudaMalloc(&da_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia_, (n_ + 1) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia_, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja_, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));

    /*
     * initialize KLU and cuSolver parameters
     */
    if(fact_ == "klu") {
      /* initialize KLU setup parameters, dont factorize yet */
      this->initializeKLU();

      /*perform KLU but only the symbolic analysis (important)   */

      klu_free_symbolic(&Symbolic_, &Common_);
      klu_free_numeric(&Numeric_, &Common_);
      Symbolic_ = klu_analyze(n_, kRowPtr_, jCol_, &Common_);

      if(Symbolic_ == nullptr) {
        nlp_->log->printf(hovError,  // catastrophic failure
                          "Symbolic factorization failed!\n");
      }
    } else { // for future
      assert(0 && "Only KLU is available for the first factorization.\n");
    }
  }
  
  

  //
  // Private functions start here
  //

  // helper private function needed for format conversion
  int hiopLinSolverSymSparseCUSOLVER::createM(const int n, 
                                              const int /* nnzL */,
                                              const int* Lp, 
                                              const int* Li,
                                              const int /* nnzU */, 
                                              const int* Up,
                                              const int* Ui)
  {
    int row;
    for(int i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        // BUT dont count diagonal twice, important
        if(row != i) {
          mia_[row + 1]++;
        }
      }
      // then each column of U
      for(int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mia_[row + 1]++;
      }
    }
    // then organize mia_;
    mia_[0] = 0;
    for(int i = 1; i < n + 1; i++) {
      mia_[i] += mia_[i - 1];
    }

    std::vector<int> Mshifts(n, 0);
    for(int i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        if(row != i) {
          // place (row, i) where it belongs!
          mja_[mia_[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      // each column of U next
      for(int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    return 0;
  }
  
  // Error checking utility for CUDA
  // KS: might later become part of src/Utils, putting it here for now
  template <typename T>
  void hiopLinSolverSymSparseCUSOLVER::hiopCheckCudaError(T result,
                                                          const char* const file,
                                                          int const line)
  {
    if(result) {
      nlp_->log->printf(hovError, "CUDA error at %s:%d, error# %d\n", file, line, result);
      assert(false);
    }
  }

  int hiopLinSolverSymSparseCUSOLVER::initializeKLU()
  {
    // KLU
    klu_defaults(&Common_);

    // TODO: consider making this a part of setup options so that user can
    // set up these values. For now, we keep them hard-wired.
    Common_.btf = 0;
    Common_.ordering = ordering_; // COLAMD=1; AMD=0
    Common_.tol = 0.1;
    Common_.scale = -1;
    Common_.halt_if_singular = 1;

    return 0;
  }

  int hiopLinSolverSymSparseCUSOLVER::initializeCusolverGLU()
  {
    nlp_->log->printf(hovScalars, "CUSOLVER: Glu \n");
    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

    // info (data structure where factorization is stored)
    //this is done in the constructor - however, this function might be called more than once
    cusolverSpDestroyGluInfo(info_M_);
    cusolverSpCreateGluInfo(&info_M_);

    return 0;
  }

  int hiopLinSolverSymSparseCUSOLVER::initializeCusolverRf()
  {
    nlp_->log->printf(hovScalars, "CUSOLVER: Rf \n");
    cusolverRfCreate(&handle_rf_);

    checkCudaErrors(cusolverRfSetAlgs(handle_rf_,
                                      CUSOLVERRF_FACTORIZATION_ALG2,
                                      CUSOLVERRF_TRIANGULAR_SOLVE_ALG2));

    checkCudaErrors(
        cusolverRfSetMatrixFormat(handle_rf_, 
                                  CUSOLVERRF_MATRIX_FORMAT_CSR,
                                  CUSOLVERRF_UNIT_DIAGONAL_STORED_L));

    cusolverRfSetResetValuesFastMode(handle_rf_,
                                     CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);

    const double boost = 1e-12;
    const double zero = 1e-14;

    cusolverRfSetNumericProperties(handle_rf_, zero, boost);
    return 0;
  }


  // call if both the matrix and the nnz structure changed or if convergence is
  // poor while using refactorization.
  int hiopLinSolverSymSparseCUSOLVER::refactorizationSetupCusolverGLU()
  {
    // for now this ONLY WORKS if proceeded by KLU. Might be worth decoupling
    // later

    // get sizes
    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;

    const int nnzM = (nnzL + nnzU - n_);

    /* parse the factorization */

    mia_ = new int[n_ + 1]{0};
    mja_ = new int[nnzM]{0};
    int* Lp = new int[n_ + 1];
    int* Li = new int[nnzL];
    // we cant use nullptr instrad od Lx and Ux because it causes SEG FAULT. It
    // seems like a waste of memory though.
  
    double* Lx = new double[nnzL];
    int* Up = new int[n_ + 1];
    int* Ui = new int[nnzU];

    double* Ux = new double[nnzU];

    int ok = klu_extract(Numeric_, 
                         Symbolic_, 
                         Lp, 
                         Li, 
                         Lx, 
                         Up, 
                         Ui, 
                         Ux, 
                         nullptr,
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr,
                         &Common_);
    createM(n_, nnzL, Lp, Li, nnzU, Up, Ui);
    
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;

    /* setup GLU */
    sp_status_ = cusolverSpDgluSetup(handle_cusolver_, 
                                     n_, nnz_, 
                                     descr_A_, 
                                     kRowPtr_,
                                     jCol_, 
                                     Numeric_->Pnum, /* base-0 */
                                     Symbolic_->Q,   /* base-0 */
                                     nnzM,           /* nnzM */
                                     descr_M_, 
                                     mia_, 
                                     mja_, 
                                     info_M_);

    sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_, info_M_, &size_M_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    buffer_size_ = size_M_;
    cudaFree(d_work_);
    checkCudaErrors(cudaMalloc((void**)&d_work_, buffer_size_));

    sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_, info_M_, d_work_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    // reset and refactor so factors are ON THE GPU

    sp_status_ = cusolverSpDgluReset(handle_cusolver_, 
                                     n_,
                                     /* A is original matrix */
                                     nnz_, 
                                     descr_A_, 
                                     da_, 
                                     dia_, 
                                     dja_, 
                                     info_M_);
 
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
    return 0;
  }
  
  int
  hiopLinSolverSymSparseCUSOLVER::refactorizationSetupCusolverRf()
  {
    // for now this ONLY WORKS if proceeded by KLU. Might be worth decoupling
    // later
    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;
    
    checkCudaErrors(cudaMalloc(&d_P, (n_) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Q, (n_) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_T, (n_) * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_P, Numeric_->Pnum, sizeof(int) * (n_), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Q, Symbolic_->Q, sizeof(int) * (n_), cudaMemcpyHostToDevice));

    int* Lp = new int[n_ + 1];
    int* Li = new int[nnzL];
    double* Lx = new double[nnzL];
    int* Up = new int[n_ + 1];
    int* Ui = new int[nnzU];
    double* Ux = new double[nnzU];

    int ok = klu_extract(Numeric_, 
                         Symbolic_, 
                         Lp, 
                         Li, 
                         Lx, 
                         Up, 
                         Ui, 
                         Ux, 
                         nullptr,
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr,
                         &Common_);

    /* CSC */
    int* d_Lp;
    int* d_Li;
    int* d_Up;
    int* d_Ui;
    double* d_Lx;
    double* d_Ux;
    /* CSR */
    int* d_Lp_csr;
    int* d_Li_csr;
    int* d_Up_csr;
    int* d_Ui_csr;
    double* d_Lx_csr;
    double* d_Ux_csr;

    /* allocate CSC */
    checkCudaErrors(cudaMalloc(&d_Lp, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Li, nnzL * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Lx, nnzL * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Up, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ui, nnzU * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ux, nnzU * sizeof(double)));
    
    /* allocate CSR */
    checkCudaErrors(cudaMalloc(&d_Lp_csr, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Li_csr, nnzL * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Lx_csr, nnzL * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Up_csr, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ui_csr, nnzU * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ux_csr, nnzU * sizeof(double)));
    
    /* copy CSC to the GPU */
    checkCudaErrors(cudaMemcpy(d_Lp, Lp, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Li, Li, sizeof(int) * (nnzL), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Lx, Lx, sizeof(double) * (nnzL), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Up, Up, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ui, Ui, sizeof(int) * (nnzU), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ux, Ux, sizeof(double) * (nnzU), cudaMemcpyHostToDevice));

    /* we dont need these any more */
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;
    
    /* now CSC to CSR using the new cuda 11 awkward way */
    size_t bufferSizeL;
    size_t bufferSizeU;

    cusparseStatus_t csp = cusparseCsr2cscEx2_bufferSize(handle_,
                                                         n_,
                                                         n_,
                                                         nnzL,
                                                         d_Lx,
                                                         d_Lp,
                                                         d_Li,
                                                         d_Lx_csr,
                                                         d_Lp_csr,
                                                         d_Li_csr,
                                                         CUDA_R_64F,
                                                         CUSPARSE_ACTION_NUMERIC,
                                                         CUSPARSE_INDEX_BASE_ZERO,
                                                         CUSPARSE_CSR2CSC_ALG1,
                                                         &bufferSizeL);

    csp = cusparseCsr2cscEx2_bufferSize(handle_, 
                                        n_, 
                                        n_, 
                                        nnzU, 
                                        d_Ux, 
                                        d_Up, 
                                        d_Ui, 
                                        d_Ux_csr, 
                                        d_Up_csr, 
                                        d_Ui_csr,
                                        CUDA_R_64F, 
                                        CUSPARSE_ACTION_NUMERIC, 
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        CUSPARSE_CSR2CSC_ALG1, 
                                        &bufferSizeU);
    /* allocate buffers */

    double* d_workL;
    double* d_workU;
    checkCudaErrors(cudaMalloc((void**)&d_workL, bufferSizeL));
    checkCudaErrors(cudaMalloc((void**)&d_workU, bufferSizeU));
    
   /* actual CSC to CSR */

    csp = cusparseCsr2cscEx2(handle_, 
                             n_, 
                             n_, 
                             nnzL, 
                             d_Lx, 
                             d_Lp, 
                             d_Li, 
                             d_Lx_csr,
                             d_Lp_csr, 
                             d_Li_csr, 
                             CUDA_R_64F,
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, 
                             d_workL);

    csp = cusparseCsr2cscEx2(handle_, 
                             n_, 
                             n_, 
                             nnzU, 
                             d_Ux, 
                             d_Up, 
                             d_Ui, 
                             d_Ux_csr,
                             d_Up_csr, 
                             d_Ui_csr, 
                             CUDA_R_64F,
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, 
                             d_workU);

    (void) csp; // mute unused variable warnings
    
    /* CSC no longer needed, nor the work arrays! */

    cudaFree(d_Lp);
    cudaFree(d_Li);
    cudaFree(d_Lx);

    cudaFree(d_Up);
    cudaFree(d_Ui);
    cudaFree(d_Ux);

    cudaFree(d_workU);
    cudaFree(d_workL);

    /* actual setup */

    sp_status_ = cusolverRfSetupDevice(n_,
                                       nnz_,
                                       dia_,
                                       dja_,
                                       da_, 
                                       nnzL,
                                       d_Lp_csr,
                                       d_Li_csr,
                                       d_Lx_csr,
                                       nnzU,
                                       d_Up_csr,
                                       d_Ui_csr,
                                       d_Ux_csr,
                                       d_P,
                                       d_Q,
                                       handle_rf_);
    cudaDeviceSynchronize();
    sp_status_ = cusolverRfAnalyze(handle_rf_);
    return 0;
  }

  ////////////////////////////////////////////
  // The Solver for Nonsymmetric KKT System //
  ////////////////////////////////////////////

  hiopLinSolverNonSymSparseCUSOLVER::hiopLinSolverNonSymSparseCUSOLVER(const int& n,
                                                                       const int& nnz,
                                                                       hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
      kRowPtr_{ nullptr },
      jCol_{ nullptr }, 
      kVal_{ nullptr },
      index_covert_CSR2Triplet_{ nullptr },
      index_covert_extra_Diag2CSR_{ nullptr }, 
      n_{ n }, 
      nnz_{ 0 },
      ordering_{ 1 },
      fact_{ "klu" },       // default
      refact_{ "glu" },     // default
      factorizationSetupSucc_{ 0 }
  {
    // handles
    cusparseCreate(&handle_);
    cusolverSpCreate(&handle_cusolver_);
    cublasCreate(&handle_cublas_);

    // descriptors
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    // Set user selected options
    std::string ordering = nlp_->options->GetString("linear_solver_sparse_ordering");
    if(ordering == "amd_ssparse") {
      ordering_ = 0;
    }
    else if(ordering == "colamd_ssparse") {
      ordering_ = 1;
    }
    else {
      nlp_->log->printf(hovWarning, 
                        "Ordering %s not compatible with cuSOLVER LU, using default ...\n",
                        ordering.c_str());
      ordering_ = 1;
    }

    fact_ = nlp_->options->GetString("cusolver_lu_factorization");
    if(fact_ != "klu") {
      nlp_->log->printf(hovWarning,
                        "Factorization %s not compatible with cuSOLVER LU, using default ...\n",
                        fact_.c_str());
      fact_ = "klu";
    }

    refact_ = nlp_->options->GetString("cusolver_lu_refactorization");
    if(refact_ != "glu" && refact_ != "rf") {
      nlp_->log->printf(hovWarning, 
                        "Refactorization %s not compatible with cuSOLVER LU, using default ...\n",
                        refact_.c_str());
      refact_ = "glu";
    }
  }

  hiopLinSolverNonSymSparseCUSOLVER::~hiopLinSolverNonSymSparseCUSOLVER()
  {
    // Delete CSR matrix on CPU
    delete[] kRowPtr_;
    delete[] jCol_;
    delete[] kVal_;

    // Delete CSR <--> triplet mappings
    delete[] index_covert_CSR2Triplet_;
    delete[] index_covert_extra_Diag2CSR_;

    // Delete CSR matrix on GPU
    cudaFree(dia_);
    cudaFree(da_);
    cudaFree(dja_);

    // Delete residual and solution vectors
    cudaFree(devr_);
    cudaFree(devx_);

    // Delete workspaces and handles
    cudaFree(d_work_);
    cusparseDestroy(handle_);
    cusolverSpDestroy(handle_cusolver_);
    cublasDestroy(handle_cublas_);
    cusparseDestroyMatDescr(descr_A_);

    if(refact_ == "glu") {
      cusparseDestroyMatDescr(descr_M_);
      cusolverSpDestroyGluInfo(info_M_);
    }

    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
    delete [] mia_;
    delete [] mja_;
  }

  int hiopLinSolverNonSymSparseCUSOLVER::matrixChanged()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if(!kRowPtr_) {
      this->firstCall();
    } else {
      // update matrix
      for(int k = 0; k < nnz_; k++) {
        kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i = 0; i < n_; i++) {
        if(index_covert_extra_Diag2CSR_[i] != -1) {
          kVal_[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
        }
      }
    } // else
    if((Numeric_ == nullptr) && (factorizationSetupSucc_ == 0)) {
      Numeric_ = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic_, &Common_);
      if(Numeric_ == nullptr) {
        nlp_->log->printf(hovWarning, "Numeric klu factorization failed. Regularizing ...\n");
        // This is not a catastrophic failure
        // The matrix is singular so return -1 to regularaize!
        return -1;
      } else { // Numeric was succesfull so now can set up
        factorizationSetupSucc_ = 1;
        nlp_->log->printf(hovScalars, "Numeric klu factorization succesful! \n");
        if(refact_ == "glu") {
          this->initializeCusolverGLU();
          this->refactorizationSetupCusolverGLU();
        } else if(refact_ == "rf") {
          this->initializeCusolverRf();
          this->refactorizationSetupCusolverRf();
        } else { // for future -
          assert(0 && "Only glu and rf refactorizations available.\n");
        }
      }
    } else { // Numeric_ != nullptr
      checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
      // re-factor here
      if(refact_ == "glu") {
        sp_status_ = cusolverSpDgluReset(handle_cusolver_, 
                                         n_,
                                         /* A is original matrix */
                                         nnz_,
                                         descr_A_,
                                         da_,
                                         dia_,
                                         dja_,
                                         info_M_);

        sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
      } else {
        if(refact_ == "rf") {
          sp_status_ = cusolverRfResetValues(n_, 
                                             nnz_, 
                                             dia_,
                                             dja_,
                                             da_,
                                             d_P,
                                             d_Q,
                                             handle_rf_);
          cudaDeviceSynchronize();
          sp_status_ = cusolverRfRefactor(handle_rf_);
        }
      }
      // end of factor
    }
    return 0;
  }

  bool
  hiopLinSolverNonSymSparseCUSOLVER::solve(hiopVector& x_)
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);
    assert(x_.get_size() == M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != nullptr);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    x->copyToDev();
    double* dx = x->local_data();
    // rhs->copyToDev();
    double* drhs = rhs->local_data();
    // double* devr_ = rhs->local_data();
    checkCudaErrors(cudaMemcpy(devr_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    // solve HERE

    if(refact_ == "glu") {
      sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                       n_,
                                       /* A is original matrix */
                                       nnz_,
                                       descr_A_,
                                       da_,
                                       dia_,
                                       dja_,
                                       devr_,/* right hand side */
                                       devx_,/* left hand side */
                                       &ite_refine_succ_,
                                       &r_nrminf_,
                                       info_M_,
                                       d_work_);
      if(sp_status_ == 0) {
        checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
      }
    } else {
      if(refact_ == "rf") {
        if(Numeric_ == nullptr) {
          sp_status_ = cusolverRfSolve(handle_rf_,
                                       d_P,
                                       d_Q,
                                       1,
                                       d_T,
                                       n_,
                                       devr_,
                                       n_);
          if(sp_status_ == 0) {
            checkCudaErrors(cudaMemcpy(dx, devr_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
          } else {
            nlp_->log->printf(hovError,  // catastrophic failure
                              "Solve failed with starus: %d\n", 
                              sp_status_);
            return false;
          }
        } else {
          memcpy(dx, drhs, sizeof(double) * n_);
          int ok = klu_solve(Symbolic_, Numeric_, n_, 1, dx, &Common_);
          klu_free_numeric(&Numeric_, &Common_);
          klu_free_symbolic(&Symbolic_, &Common_);
        }
      } else {
        nlp_->log->printf(hovError, // catastrophic failure
                          "Unknown refactorization, exiting\n");
        assert(false && "Only GLU and cuSolverRf are available refactorizations.");
      }
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    delete rhs;
    rhs = nullptr;
    return true;
  }

  void
  hiopLinSolverNonSymSparseCUSOLVER::firstCall()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the
    // sparse matrix, and the 2nd part are the additional dia_gonal elememts
    // the 1st part is sorted by row
    hiop::hiopMatrixSparseTriplet* M_triplet = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(M_);

    if(M_triplet == nullptr) {
      nlp_->log->printf(hovError, "M_triplet is nullptr");
      return;
    }

    M_triplet->convertToCSR(nnz_,
                            &kRowPtr_,
                            &jCol_,
                            &kVal_,
                            &index_covert_CSR2Triplet_,
                            &index_covert_extra_Diag2CSR_,
                            extra_dia_g_nnz_map);

    /*
     * initialize cusolver parameters
     */


    // allocate gpu data
    cudaFree(devx_);
    cudaFree(devr_);
    devx_ = nullptr;
    devr_ = nullptr;
    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));

    cudaFree(dia_);
    cudaFree(da_);
    cudaFree(dja_);
    /* this has to be done no matter what */

    checkCudaErrors(cudaMalloc(&da_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia_, (n_ + 1) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia_, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja_, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
    /*
     * initialize KLU and cuSolver parameters
     */

    if(fact_ == "klu") {
      /* initialize KLU setup parameters, dont factorize yet */
      this->initializeKLU();

      /*perform KLU but only the symbolic analysis (important)   */

      klu_free_symbolic(&Symbolic_, &Common_);
      klu_free_numeric(&Numeric_, &Common_);
      Symbolic_ = klu_analyze(n_, kRowPtr_, jCol_, &Common_);

      if(Symbolic_ == nullptr) {
        nlp_->log->printf(hovError,
                          "symbolic nullptr\n"); // catastrophic failure
      }
    } else { // for future
      assert(0 && "Only KLU initial factorization is available.\n");
    }
  }
  
  //  
  // Private functions start here
  //

  // helper private function needed for format conversion
  int hiopLinSolverNonSymSparseCUSOLVER::createM(const int n, 
                                                 const int /* nnzL */,
                                                 const int* Lp, 
                                                 const int* Li,
                                                 const int /* nnzU */, 
                                                 const int* Up,
                                                 const int* Ui)
  {
    int row;
    for(int i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        // BUT dont count dia_gonal twice, important
        if(row != i) {
          mia_[row + 1]++;
        }
      }
      // then each column of U
      for(int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mia_[row + 1]++;
      }
    }
    // then organize mia_;
    mia_[0] = 0;
    for(int i = 1; i < n + 1; ++i) {
      mia_[i] += mia_[i - 1];
    }

    int* Mshifts = new int[n]{0};
    for(int i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        if(row != i) {
          // place (row, i) where it belongs!
          mja_[mia_[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      // each column of U next
      for(int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    delete [] Mshifts;
    return 0;
  }

  template <typename T>
  void
  hiopLinSolverNonSymSparseCUSOLVER::hiopCheckCudaError(T result,
                                                        const char* const file,
                                                        int const line)
  {
    if(result) {
      nlp_->log->printf(hovError, "CUDA error at %s:%d, error# %d\n", file, line, result);
      exit(EXIT_FAILURE);
    }
  }

  int
  hiopLinSolverNonSymSparseCUSOLVER::initializeKLU()
  {
    klu_defaults(&Common_);

    // TODO: consider making a part of setup options that can be called from a
    // user side For now, keeping these options hard-wired
    Common_.btf = 0;
    Common_.ordering = ordering_; // COLAMD=1; AMD=0
    Common_.tol = 0.01;
    Common_.scale = -1;
    Common_.halt_if_singular = 1;

    return 0;
  }

  int
  hiopLinSolverNonSymSparseCUSOLVER::initializeCusolverGLU()
  {

    nlp_->log->printf(hovScalars, "CUSOLVER: Glu \n");
    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

    // info (data structure where factorization is stored)
    cusolverSpCreateGluInfo(&info_M_);

    return 0;
  }

  int
  hiopLinSolverNonSymSparseCUSOLVER::initializeCusolverRf()
  {
    nlp_->log->printf(hovScalars, "CUSOLVER: Rf \n");
    cusolverRfCreate(&handle_rf_);

    checkCudaErrors(cusolverRfSetAlgs(handle_rf_,
                                      CUSOLVERRF_FACTORIZATION_ALG2,
                                      CUSOLVERRF_TRIANGULAR_SOLVE_ALG2));

    checkCudaErrors(cusolverRfSetMatrixFormat(handle_rf_, 
                                              CUSOLVERRF_MATRIX_FORMAT_CSR, 
                                              CUSOLVERRF_UNIT_DIAGONAL_STORED_L));

    cusolverRfSetResetValuesFastMode(handle_rf_,
                                     CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);

    const double boost = 1e-12;
    const double zero = 1e-14;

    cusolverRfSetNumericProperties(handle_rf_, zero, boost);
    return 0;
  }

  int
  hiopLinSolverNonSymSparseCUSOLVER::refactorizationSetupCusolverGLU()
  {
    // get sizes

    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;

    const int nnzM = (nnzL + nnzU - n_);
    /* parse the factorization */

    mia_ = new int[n_ + 1]{0};
    mja_ = new int[nnzM]{0};

    int* Lp = new int[n_ + 1];
    int* Li = new int[nnzL];
    // we cant use nullptr instrad od Lx and Ux because it causes SEG FAULT. It
    // seems like a waste of memory though.
    double* Lx = new double[nnzL];
    int* Up = new int[n_ + 1];
    int* Ui = new int[nnzU];
    double* Ux = new double[nnzU];
    int ok = klu_extract(Numeric_, 
                         Symbolic_, 
                         Lp, 
                         Li, 
                         Lx, 
                         Up, 
                         Ui, 
                         Ux, 
                         nullptr,
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr,
                         &Common_);
    createM(n_, nnzL, Lp, Li, nnzU, Up, Ui);
    
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;

    /* setup GLU */
    sp_status_ = cusolverSpDgluSetup(handle_cusolver_, 
                                     n_, nnz_, 
                                     descr_A_, 
                                     kRowPtr_,
                                     jCol_, 
                                     Numeric_->Pnum, /* base-0 */
                                     Symbolic_->Q,          /* base-0 */
                                     nnzM,                  /* nnzM */
                                     descr_M_, 
                                     mia_, 
                                     mja_, 
                                     info_M_);


    sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_, info_M_, &size_M_);

    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    buffer_size_ = size_M_;
    cudaFree(d_work_);
    checkCudaErrors(cudaMalloc((void**)&d_work_, buffer_size_));

    sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_, info_M_, d_work_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    // reset and refactor so factors are ON THE GPU

    sp_status_ = cusolverSpDgluReset(handle_cusolver_, 
                                     n_,
                                     /* A is original matrix */
                                     nnz_, 
                                     descr_A_, 
                                     da_, 
                                     dia_, 
                                     dja_, 
                                     info_M_);
    
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
    return 0;
  }

  int hiopLinSolverNonSymSparseCUSOLVER::refactorizationSetupCusolverRf()
  {
    // for now this ONLY WORKS if proceeded by KLU. Might be worth decoupling
    // later
    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;
    
    checkCudaErrors(cudaMalloc(&d_P, (n_) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Q, (n_) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_T, (n_) * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_P, Numeric_->Pnum, sizeof(int) * (n_), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Q, Symbolic_->Q, sizeof(int) * (n_), cudaMemcpyHostToDevice));

    int* Lp = new int[n_ + 1];
    int* Li = new int[nnzL];
    double* Lx = new double[nnzL];
    int* Up = new int[n_ + 1];
    int* Ui = new int[nnzU];
    double* Ux = new double[nnzU];

    int ok = klu_extract(Numeric_, 
                         Symbolic_, 
                         Lp, 
                         Li, 
                         Lx, 
                         Up, 
                         Ui, 
                         Ux, 
                         nullptr,
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr, 
                         nullptr,
                         &Common_);

    /* CSC */
    int* d_Lp;
    int* d_Li;
    int* d_Up;
    int* d_Ui;
    double* d_Lx;
    double* d_Ux;
    /* CSR */
    int* d_Lp_csr;
    int* d_Li_csr;
    int* d_Up_csr;
    int* d_Ui_csr;
    double* d_Lx_csr;
    double* d_Ux_csr;

    /* allocate CSC */
    checkCudaErrors(cudaMalloc(&d_Lp, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Li, nnzL * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Lx, nnzL * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Up, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ui, nnzU * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ux, nnzU * sizeof(double)));
    
    /* allocate CSR */
    checkCudaErrors(cudaMalloc(&d_Lp_csr, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Li_csr, nnzL * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Lx_csr, nnzL * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Up_csr, (n_ + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ui_csr, nnzU * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Ux_csr, nnzU * sizeof(double)));
    
    /* copy CSC to the GPU */
    checkCudaErrors(cudaMemcpy(d_Lp, Lp, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Li, Li, sizeof(int) * (nnzL), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Lx, Lx, sizeof(double) * (nnzL), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Up, Up, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ui, Ui, sizeof(int) * (nnzU), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ux, Ux, sizeof(double) * (nnzU), cudaMemcpyHostToDevice));

    /* we dont need these any more */
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;
    
    /* now CSC to CSR using the new cuda 11 awkward way */
    size_t bufferSizeL, bufferSizeU;
    
    cusparseStatus_t csp = cusparseCsr2cscEx2_bufferSize(handle_,
                                                         n_,
                                                         n_,
                                                         nnzL,
                                                         d_Lx,
                                                         d_Lp,
                                                         d_Li,
                                                         d_Lx_csr,
                                                         d_Lp_csr,
                                                         d_Li_csr,
                                                         CUDA_R_64F,
                                                         CUSPARSE_ACTION_NUMERIC,
                                                         CUSPARSE_INDEX_BASE_ZERO,
                                                         CUSPARSE_CSR2CSC_ALG1,
                                                         &bufferSizeL);

    csp = cusparseCsr2cscEx2_bufferSize(handle_, 
                                        n_, 
                                        n_, 
                                        nnzU, 
                                        d_Ux, 
                                        d_Up, 
                                        d_Ui, 
                                        d_Ux_csr, 
                                        d_Up_csr, 
                                        d_Ui_csr,
                                        CUDA_R_64F, 
                                        CUSPARSE_ACTION_NUMERIC, 
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        CUSPARSE_CSR2CSC_ALG1, 
                                        &bufferSizeU);
    /* allocate buffers */
    double* d_workL;
    double* d_workU;
    checkCudaErrors(cudaMalloc((void**)&d_workL, bufferSizeL));
    checkCudaErrors(cudaMalloc((void**)&d_workU, bufferSizeU));
    
    /* actual CSC to CSR */
    csp = cusparseCsr2cscEx2(handle_, 
                             n_, 
                             n_, 
                             nnzL, 
                             d_Lx, 
                             d_Lp, 
                             d_Li, 
                             d_Lx_csr,
                             d_Lp_csr, 
                             d_Li_csr, 
                             CUDA_R_64F,
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, 
                             d_workL);

    csp = cusparseCsr2cscEx2(handle_, 
                             n_, 
                             n_, 
                             nnzU, 
                             d_Ux, 
                             d_Up, 
                             d_Ui, 
                             d_Ux_csr,
                             d_Up_csr, 
                             d_Ui_csr, 
                             CUDA_R_64F,
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, 
                             d_workU);
    
    (void) csp; // silence warnings for unused csp

    /* CSC no longer needed, nor the work arrays! */

    cudaFree(d_Lp);
    cudaFree(d_Li);
    cudaFree(d_Lx);

    cudaFree(d_Up);
    cudaFree(d_Ui);
    cudaFree(d_Ux);

    cudaFree(d_workU);
    cudaFree(d_workL);

    /* actual setup */

    sp_status_ = cusolverRfSetupDevice(n_,
                                       nnz_,
                                       dia_,
                                       dja_,
                                       da_, 
                                       nnzL,
                                       d_Lp_csr,
                                       d_Li_csr,
                                       d_Lx_csr,
                                       nnzU,
                                       d_Up_csr,
                                       d_Ui_csr,
                                       d_Ux_csr,
                                       d_P,
                                       d_Q,
                                       handle_rf_);
    cudaDeviceSynchronize();
    sp_status_ = cusolverRfAnalyze(handle_rf_);
    return 0;
  }

} // namespace hiop
