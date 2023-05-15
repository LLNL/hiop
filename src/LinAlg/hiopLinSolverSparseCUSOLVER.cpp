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
#include "KrylovSolverKernels.h"

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
    fact_{ "klu" }, // default
    refact_{ "glu" }, // default
    factorizationSetupSucc_{ 0 },
    is_first_solve_{ true },
    is_first_call_{ true }
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
    // by default, dont use iterative refinement
    int maxit_test  = nlp_->options->GetInteger("ir_inner_maxit");

    if ((maxit_test < 0) || (maxit_test > 1000)){
      nlp_->log->printf(hovWarning, 
                        "Wrong maxit value: %d. Use int maxit value between 0 and 1000. Setting default (50)  ...\n",
                        ir_->maxit_);
      maxit_test = 50;
    }
    use_ir_ = "no";
    if(maxit_test > 0){
      use_ir_ = "yes";
      ir_ = new hiopLinSolverSymSparseCUSOLVERInnerIR;
      ir_->maxit_ = maxit_test;
    } 
    if(use_ir_ == "yes") {
      if((refact_ == "rf")) {

        ir_->restart_ =  nlp_->options->GetInteger("ir_inner_restart");

        if ((ir_->restart_ <0) || (ir_->restart_ >100)){
          nlp_->log->printf(hovWarning, 
                            "Wrong restart value: %d. Use int restart value between 1 and 100. Setting default (20)  ...\n",
                            ir_->restart_);
          ir_->restart_ = 20;
        }


        ir_->tol_  = nlp_->options->GetNumeric("ir_inner_tol");
        if ((ir_->tol_ <0) || (ir_->tol_ >1)){
          nlp_->log->printf(hovWarning, 
                            "Wrong tol value: %e. Use double tol value between 0 and 1. Setting default (1e-12)  ...\n",
                            ir_->tol_);
          ir_->tol_ = 1e-12;
        }
        ir_->orth_option_ = nlp_->options->GetString("ir_inner_cusolver_gs_scheme");

        /* 0) "Standard" GMRES and FGMRES (Saad and Schultz, 1986, Saad, 1992) use Modified Gram-Schmidt ("mgs") to keep the Krylov vectors orthogonal. 
         * Modified Gram-Schmidt requires k synchronization (due to inner products) in iteration k and this becomes a scaling bottleneck for 
         * GPU-accelerated implementation and it becomes even more pronouced for MPI+GPU-acceleration.
         * Modified Gram-Schidt can be replaced by a different scheme.
         *
         * 1) One can use Classical Gram-Schmidt ("cgs") which is numerically unstable or reorthogonalized Classical Gram-Schmidt ("cgs2"), which
         * is numerically stable and requires 3 synchrnozations and each iteration. Reorthogonalized Classical Gram-Schmidt makes two passes of
         * Classical Gram-Schmidt. And two passes are enough to get vectors orthogonal to machine precision (Bjorck 1967).
         * 
         * 2) An alternative is a low-sych version (Swirydowicz and Thomas, 2020), which reformulates Modified Gram-Schmidt to be a (very small) triangular solve.
         * It requires extra storage for the matrix used in triangular solve (kxk at iteration k), but only two sycnhronizations are needed per iteration.
         * The inner producats are performed in bulk, which quarantees better GPU utilization. The second synchronization comes from normalizing the vector and 
         * can be eliminated if the norm is postponed to the next iteration, but also makes code more complicated. This is why we use two-synch method ("mgs_two_synch")
         * 
         * 3) A recently submitted paper by Stephen Thomas (Thomas 202*) takes the triangular solve idea further and uses a different approximation for 
         * the inverse of a triangular matrix. It requires two (very small) triangular solves and two sychroniztions (if the norm is NOT delayed). It also guarantees 
         * that the vectors are orthogonal to the machine epsilon, as in cgs2. Since Stephe's paper is named "post modern GMRES", we call this Gram-Schmidt scheme "mgs_pm".
         */ 
        if(ir_->orth_option_ != "mgs" && ir_->orth_option_ != "cgs2" && ir_->orth_option_ != "mgs_two_synch" && ir_->orth_option_ != "mgs_pm") {
          nlp_->log->printf(hovWarning, 
                            "mgs option : %s is wrong. Use 'mgs', 'cgs2', 'mgs_two_synch' or 'mgs_pm'. Switching to default (mgs) ...\n",
                            use_ir_.c_str());
          ir_->orth_option_ = "mgs";
        }

        ir_->conv_cond_ =  nlp_->options->GetInteger("ir_inner_conv_cond");

        if ((ir_->conv_cond_ <0) || (ir_->conv_cond_ >2)){
          nlp_->log->printf(hovWarning, 
                            "Wrong IR convergence condition: %d. Use int value: 0, 1 or 2. Setting default (0)  ...\n",
                            ir_->conv_cond_);
          ir_->conv_cond_ = 0;
        }

      } else {
        nlp_->log->printf(hovWarning, 
                          "Currently, inner iterative refinement works ONLY with cuSolverRf ... \n");
        use_ir_ = "no";
      }
    }
  } // constructor

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

    // Delete `matrix descriptor used in cuSolverGLU setup
    if(refact_ == "glu") {
      cusparseDestroyMatDescr(descr_M_);
      cusolverSpDestroyGluInfo(info_M_);
    }

    if(refact_ == "rf") {
      cudaFree(d_P_);
      cudaFree(d_Q_);
      cudaFree(d_T_);
    }
    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
    delete [] mia_;
    delete [] mja_;

    // Experimental code: delete IR
    if(use_ir_ == "yes") {
      // destroy IR object
      delete ir_;
    }
    // End of experimetnal code
  }

  int hiopLinSolverSymSparseCUSOLVER::matrixChanged()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    nlp_->runStats.linsolv.tmFactTime.start();

    // if(!kRowPtr_) {
    if(is_first_call_) {
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
    } // else

    if(/*(Numeric_ == nullptr) &&*/ (factorizationSetupSucc_ == 0)) {
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
          if(use_ir_ == "yes") {
            this->IRsetup();
          }
        } else { // for future -
          assert(0 && "Only glu and rf refactorizations available.\n");
        }
      }
    } else { // Numeric_ != nullptr OR factorizationSetupSucc_ == 1
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
                                             d_P_,
                                             d_Q_,
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

    // Set IR tolerance to be `factor*mu` and no less than `mintol`
    double factor = nlp_->options->GetNumeric("ir_inner_cusolver_tol_factor");
    double mintol = nlp_->options->GetNumeric("ir_inner_cusolver_tol_min");
    double ir_tol = std::min(factor*(nlp_->mu()), mintol);
    nlp_->log->printf(hovScalars,
                      "Barrier parameter mu = %g, IR tolerance set to %g.\n", nlp_->mu(), ir_tol);
    // // Debugging output
    // std::cout << "mu in cusolver = " <<  nlp_->mu() << "\n";
    // std::cout << "ir tol         = " <<  ir_tol << "\n";

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
                          "Solve failed with status: %d\n", 
                          sp_status_);
        return false;
      }
    } else {
      if(refact_ == "rf") {
        // if(Numeric_ == nullptr) {
        if(!is_first_solve_) {
          sp_status_ = cusolverRfSolve(handle_rf_,
                                       d_P_,
                                       d_Q_,
                                       1,
                                       d_T_,
                                       n_,
                                       devr_,
                                       n_);

          if(sp_status_ == 0) {
            // Experimental code for IR
            if(use_ir_ == "yes") {
              // Set tolerance based on barrier parameter mu
              ir_->set_tol(ir_tol);
              nlp_->log->printf(hovScalars,
                                "Running iterative refinement with tol %e\n", ir_tol);
              checkCudaErrors(cudaMemcpy(devx_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));
              //experimental code 

              //end of experimental code
              ir_->fgmres(devr_, devx_);             

              nlp_->log->printf(hovScalars, 
                                "\t fgmres: init residual norm  %e final residual norm %e number of iterations %d\n", 
                                ir_->getInitialResidalNorm(), 
                                ir_->getFinalResidalNorm(), 
                                ir_->getFinalNumberOfIterations());
            }
            // End of Experimental code
            checkCudaErrors(cudaMemcpy(dx, devr_, sizeof(double) * n_, cudaMemcpyDeviceToHost));


          } else {
            nlp_->log->printf(hovError,  // catastrophic failure
                              "Solve failed with status: %d\n", 
                              sp_status_);
            return false;
          }
        } else {
          memcpy(dx, drhs, sizeof(double) * n_);
          int ok = klu_solve(Symbolic_, Numeric_, n_, 1, dx, &Common_);
          klu_free_numeric(&Numeric_, &Common_);
          klu_free_symbolic(&Symbolic_, &Common_);
          is_first_solve_ = false;
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

    // Transfer triplet to CSR form
    // Allocate row pointers and compute number of nonzeros.
    kRowPtr_ = new int[n_ + 1]{ 0 };
    compute_nnz();
    // Set column indices and matrix values.
    kVal_ = new double[nnz_]{ 0.0 };
    jCol_ = new int[nnz_]{ 0 };
    set_csr_indices_values();

    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));

    checkCudaErrors(cudaMalloc(&da_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia_, (n_ + 1) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia_, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja_, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
    if (use_ir_ == "yes"){
      cusparseCreateCsr(&(ir_->mat_A_), 
                        n_, 
                        n_, 
                        nnz_,
                        dia_, 
                        dja_, 
                        da_,
                        CUSPARSE_INDEX_32I, 
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F);
    }
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
    is_first_call_ = false;
  }

  void hiopLinSolverSymSparseCUSOLVER::compute_nnz()
  {
    //
    // compute nnz in each row
    //
    // off-diagonal part
    kRowPtr_[0] = 0;
    for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
      if(M_->i_row()[k] != M_->j_col()[k]) {
        kRowPtr_[M_->i_row()[k] + 1]++;
        kRowPtr_[M_->j_col()[k] + 1]++;
        nnz_ += 2;
      }
    }
    // diagonal part
    for(int i = 0; i < n_; i++) {
      kRowPtr_[i + 1]++;
      nnz_ += 1;
    }
    // get correct row ptr index
    for(int i = 1; i < n_ + 1; i++) {
      kRowPtr_[i] += kRowPtr_[i - 1];
    }
    assert(nnz_ == kRowPtr_[n_]);
  }

  void hiopLinSolverSymSparseCUSOLVER::set_csr_indices_values()
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
        nlp_->log->printf(hovError, 
                          "CUDA error at %s:%d, error# %d\n", 
                          file, 
                          line, 
                          result);
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
                                     n_,
                                     nnz_, 
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
      // for now this ONLY WORKS if preceeded by KLU. Might be worth decoupling
      // later
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      checkCudaErrors(cudaMalloc(&d_P_, (n_) * sizeof(int)));
      checkCudaErrors(cudaMalloc(&d_Q_, (n_) * sizeof(int)));
      checkCudaErrors(cudaMalloc(&d_T_, (n_) * sizeof(double)));

      checkCudaErrors(cudaMemcpy(d_P_, Numeric_->Pnum, sizeof(int) * (n_), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(d_Q_, Symbolic_->Q, sizeof(int) * (n_), cudaMemcpyHostToDevice));

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

      (void)csp; // mute unused variable warnings

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
                                         d_P_,
                                         d_Q_,
                                         handle_rf_);
      cudaDeviceSynchronize();
      sp_status_ = cusolverRfAnalyze(handle_rf_);

      //clean up 
      cudaFree(d_Lp_csr);
      cudaFree(d_Li_csr);
      cudaFree(d_Lx_csr);

      cudaFree(d_Up_csr);
      cudaFree(d_Ui_csr);
      cudaFree(d_Ux_csr);

      return 0;
    }

  // Experimental: setup the iterative refinement
  void hiopLinSolverSymSparseCUSOLVER::IRsetup()
  {
    ir_->cusparse_handle_ = handle_;
    ir_->cublas_handle_ = handle_cublas_;
    ir_->cusolverrf_handle_ = handle_rf_;
    ir_->cusolver_handle_ = handle_cusolver_;
    ir_->n_ = n_;
    ir_->nnz_ = nnz_;

    ir_->dia_ = dia_;
    ir_->da_ = da_;
    // only set pointers
    ir_->d_T_ = d_T_;
    ir_->d_P_ = d_P_;
    ir_->d_Q_ = d_Q_;

    // setup matvec

    cusparseCreateDnVec(&ir_->vec_x_, n_, devx_, CUDA_R_64F);
    cusparseCreateDnVec(&ir_->vec_Ax_, n_, devr_, CUDA_R_64F);
    size_t buffer_size;
    checkCudaErrors(cusparseSpMV_bufferSize(ir_->cusparse_handle_, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                            &(ir_->minusone_),
                                            ir_->mat_A_,
                                            ir_->vec_x_,
                                            &(ir_->one_),
                                            ir_->vec_Ax_,
                                            CUDA_R_64F,
                                            CUSPARSE_SPMV_CSR_ALG2,
                                            &buffer_size));

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMalloc(&ir_->mv_buffer_, buffer_size));

    // allocate space for the GPU

    checkCudaErrors(cudaMalloc(&(ir_->d_V_), n_ * (ir_->restart_ + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&(ir_->d_Z_), n_ * (ir_->restart_ + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&(ir_->d_rvGPU_), 2 * (ir_->restart_ + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&(ir_->d_Hcolumn_), 2 * (ir_->restart_ + 1) * (ir_->restart_ + 1) * sizeof(double)));

    // and for the CPU

    ir_->h_H_ = new double[ir_->restart_ * (ir_->restart_ + 1)];
    ir_->h_c_ = new double[ir_->restart_];      // needed for givens
    ir_->h_s_ = new double[ir_->restart_];      // same
    ir_->h_rs_ = new double[ir_->restart_ + 1]; // for residual norm history

    // for specific orthogonalization options, need a little more memory
    if(ir_->orth_option_ == "mgs_two_synch" || ir_->orth_option_ == "mgs_pm") {
      ir_->h_L_ = new double[ir_->restart_ * (ir_->restart_ + 1)];
      ir_->h_rv_ = new double[ir_->restart_ + 1];
    }

    if(ir_->orth_option_ == "cgs2") {
      ir_->h_aux_ = new double[ir_->restart_ + 1];
      checkCudaErrors(cudaMalloc(&(ir_->d_H_col_), (ir_->restart_ + 1) * sizeof(double)));
    }

    if(ir_->orth_option_ == "mgs_pm") {
      ir_->h_aux_ = new double[ir_->restart_ + 1];
    }
  }
  // Experimental code ends here

  ////////////////////////////////////////////////////////
  // Inner iterative refinement class methods
  ////////////////////////////////////////////////////////

  // Default constructor
  hiopLinSolverSymSparseCUSOLVERInnerIR::hiopLinSolverSymSparseCUSOLVERInnerIR()
  {}

  // Parametrized constructor
  hiopLinSolverSymSparseCUSOLVERInnerIR::hiopLinSolverSymSparseCUSOLVERInnerIR(int restart, 
                                                                               double tol, 

                                                                               int maxit)
    : restart_{restart}, 
    maxit_{maxit},
    tol_{tol}
  {}

  hiopLinSolverSymSparseCUSOLVERInnerIR::~hiopLinSolverSymSparseCUSOLVERInnerIR()
  {
    cusparseDestroySpMat(mat_A_);
    // free GPU variables that belong to this class and are not shared with CUSOLVER class
    cudaFree(mv_buffer_);
    cudaFree(d_V_);
    cudaFree(d_Z_);
    cudaFree(d_rvGPU_);
    cudaFree(d_Hcolumn_);

    if(orth_option_ == "cgs2") {
      cudaFree(d_H_col_);
    }
    // delete all CPU GMRES variables
    delete[] h_H_;

    if(orth_option_ == "mgs_two_synch" || orth_option_ == "mgs_pm") {
      delete[] h_L_;
      delete[] h_rv_;
    }
    delete[] h_c_;
    delete[] h_s_;
    delete[] h_rs_;

    if(orth_option_ == "mgs_pm" || orth_option_ == "cgs2") {
      delete[] h_aux_;
    }
  }

  double hiopLinSolverSymSparseCUSOLVERInnerIR::getFinalResidalNorm()
  {
    return final_residual_norm_;
  }

  double hiopLinSolverSymSparseCUSOLVERInnerIR::getInitialResidalNorm()
  {
    return initial_residual_norm_;
  }

  int hiopLinSolverSymSparseCUSOLVERInnerIR::getFinalNumberOfIterations()
  {
    return fgmres_iters_;
  }


  double hiopLinSolverSymSparseCUSOLVERInnerIR::matrixAInfNrm()
  {
    double nrm;
    matrix_row_sums(n_, nnz_, dia_, da_, d_Z_); 
    cusolverSpDnrminf(cusolver_handle_,
                      n_,
                      d_Z_,
                      &nrm,
                      mv_buffer_  /* at least 8192 bytes */);
    return nrm;
  }

  double hiopLinSolverSymSparseCUSOLVERInnerIR::vectorInfNrm(int n, double* d_v)
  {
    double nrm; 

    cusolverSpDnrminf(cusolver_handle_,
                      n,
                      d_v,
                      &nrm,
                      mv_buffer_  /* at least 8192 bytes */);
    return nrm;
  }

  void hiopLinSolverSymSparseCUSOLVERInnerIR::fgmres(double *d_x, double *d_b)
  {
    int outer_flag = 1;
    int notconv = 1; 
    int i = 0;
    int it = 0;
    int j;
    int k;
    int k1;

    double t;
    double rnorm;
    double bnorm;
    // double rnorm_aux;
    double tolrel;
    //V[0] = b-A*x_0
    cudaMemcpy(&(d_V_[0]), d_b, sizeof(double) * n_, cudaMemcpyDeviceToDevice);

    cudaMatvec(d_x, d_V_, "residual");

    rnorm = 0.0;
    cublasDdot (cublas_handle_,  n_, d_b, 1, d_b, 1, &bnorm);
    cublasDdot (cublas_handle_,  n_, d_V_, 1, d_V_, 1, &rnorm);
    //rnorm = ||V_1||
    rnorm = sqrt(rnorm);
    bnorm = sqrt(bnorm);
    while(outer_flag) {
      // check if maybe residual is already small enough?
      if(it == 0) {
        tolrel = tol_ * rnorm;
        if(fabs(tolrel) < 1e-16) {
          tolrel = 1e-16;
        }
      }
      int exit_cond = 0;
      if (conv_cond_ == 0){
        exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON));
      } else {
        if (conv_cond_ == 1){
          exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < tol_));
        } else {
          if (conv_cond_ == 2){
            exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < (tol_*bnorm)));
          }
        }
      }
      if (exit_cond) {
        outer_flag = 0;
        final_residual_norm_ = rnorm;
        initial_residual_norm_ = rnorm;
        fgmres_iters_ = 0;
        break;
      }

      // normalize first vector
      t = 1.0 / rnorm;
      cublasDscal(cublas_handle_, n_, &t, d_V_, 1);

      // initialize norm history

      h_rs_[0] = rnorm;
      initial_residual_norm_ = rnorm;
      i = -1;
      notconv = 1;

      while((notconv) && (it < maxit_)) {
        i++;
        it++;
        // Z_i = (LU)^{-1}*V_i
        cudaMemcpy(&d_Z_[i * n_], &d_V_[i * n_], sizeof(double) * n_, cudaMemcpyDeviceToDevice);
        cusolverRfSolve(cusolverrf_handle_, d_P_, d_Q_, 1, d_T_, n_, &d_Z_[i * n_], n_);
        cudaDeviceSynchronize();
        // V_{i+1}=A*Z_i
        cudaMatvec(&d_Z_[i * n_], &d_V_[(i + 1) * n_], "matvec");
        // orthogonalize V[i+1], form a column of h_L
        GramSchmidt(i);

        if(i != 0) {
          for(int k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0

        double Hii = h_H_[i * (restart_ + 1) + i];
        double Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        double gam = sqrt(Hii * Hii + Hii1 * Hii1);

        if(fabs(gam - ZERO) <= EPSILON) {
          gam = EPSMAC;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)] = h_c_[i] * Hii + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

        // residual norm estimate
        rnorm = fabs(h_rs_[i + 1]);
        // check convergence
        if(i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_) {
          notconv = 0;
        }
      } // inner while

      // solve tri system
      h_rs_[i] = h_rs_[i] / h_H_[i * (restart_ + 1) + i];
      for(int ii = 2; ii <= i + 1; ii++) {
        k = i - ii + 1;
        k1 = k + 1;
        t = h_rs_[k];
        for(j = k1; j <= i; j++) {
          t -= h_H_[j * (restart_ + 1) + k] * h_rs_[j];
        }
        h_rs_[k] = t / h_H_[k * (restart_ + 1) + k];
      }

      // get solution
      for(j = 0; j <= i; j++) {
        cublasDaxpy(cublas_handle_, n_, &h_rs_[j], &d_Z_[j * n_], 1, d_x, 1);
      }

      /* test solution */

      if(rnorm <= tolrel || it >= maxit_) {
        // rnorm_aux = rnorm;
        outer_flag = 0;
      }

      cudaMemcpy(&d_V_[0], d_b, sizeof(double)*n_, cudaMemcpyDeviceToDevice);
      cudaMatvec(d_x, d_V_, "residual");

      rnorm = 0.0;
      cublasDdot(cublas_handle_, n_, d_V_, 1, d_V_, 1, &rnorm);
      // rnorm = ||V_1||
      rnorm = sqrt(rnorm);

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        fgmres_iters_ = it;
      }
    } // outer while
  }

  //b-Ax
  void hiopLinSolverSymSparseCUSOLVERInnerIR::cudaMatvec(double *d_x, double * d_b, std::string option)
  {
    cusparseCreateDnVec(&vec_x_, n_, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vec_Ax_, n_, d_b, CUDA_R_64F);
    if (option == "residual"){
      //b = b-Ax
      cusparseSpMV(cusparse_handle_, 
                   CUSPARSE_OPERATION_NON_TRANSPOSE,       
                   &minusone_,
                   mat_A_,
                   vec_x_, 
                   &one_, 
                   vec_Ax_, 
                   CUDA_R_64F,
                   CUSPARSE_SPMV_CSR_ALG2,
                   mv_buffer_);
    } else {
      // just b = A*x
      cusparseSpMV(cusparse_handle_, 
                   CUSPARSE_OPERATION_NON_TRANSPOSE, 
                   &one_, 
                   mat_A_, 
                   vec_x_, 
                   &zero_, 
                   vec_Ax_, 
                   CUDA_R_64F,
                   CUSPARSE_SPMV_CSR_ALG2, 
                   mv_buffer_);
    }
    cusparseDestroyDnVec(vec_x_);
    cusparseDestroyDnVec(vec_Ax_);
  }

  void hiopLinSolverSymSparseCUSOLVERInnerIR::GramSchmidt(int i)
  {
    double t;
    const double one = 1.0;
    const double minusone = -1.0;
    const double zero = 0.0;
    double s;
    int sw = 0;
    if(orth_option_ == "mgs") {
      sw = 0;
    } else {
      if(orth_option_ == "cgs2") {
        sw = 1;
      } else {
        if(orth_option_ == "mgs_two_synch") {
          sw = 2;
        } else {
          if(orth_option_ == "mgs_pm") {
            sw = 3;
          } else {
            // display error message and set sw = 0;
            /*
               nlp_->log->printf(hovWarning, 
               "Wrong Gram-Schmidt option. Setting default (modified Gram-Schmidt, mgs) ...\n");
               */
            sw = 0;
          }
        }
      }
    }

    switch (sw){
      case 0: //mgs

        for(int j=0; j<=i; ++j) {
          t=0.0;
          cublasDdot (cublas_handle_,  n_, &d_V_[j*n_], 1, &d_V_[(i+1)*n_], 1, &t);

          h_H_[i*(restart_+1)+j]=t; 
          t *= -1.0;

          cublasDaxpy(cublas_handle_,
                      n_,
                      &t,
                      &d_V_[j*n_],
                      1,
                      &d_V_[(i+1)*n_],
                      1);
        }
        t = 0.0;
        cublasDdot(cublas_handle_,  n_, &d_V_[(i+1)*n_], 1, &d_V_[(i+1)*n_], 1, &t);

        //set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i)*(restart_+1)+i+1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1); 
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
        }
        break;

      case 1://cgs2
        // Hcol = V(:,1:i)^T *V(:,i+1);
        cublasDgemv(cublas_handle_,
                    CUBLAS_OP_T,
                    n_,
                    i+1,
                    &one_,
                    d_V_,
                    n_,
                    &d_V_[(i+1)*n_],
                    1,
                    &zero_,d_H_col_,
                    1);
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
        cublasDgemv(cublas_handle_,
                    CUBLAS_OP_N,
                    n_,
                    i+1,
                    &minusone_,
                    d_V_,
                    n_,
                    d_H_col_,
                    1,
                    &one_,
                    &d_V_[n_*(i+1)],
                    1);
        // copy H_col to aux, we will need it later

        cudaMemcpy(h_aux_, d_H_col_, sizeof(double) * (i+1), cudaMemcpyDeviceToHost);

        //Hcol = V(:,1:i)*V(:,i+1);
        cublasDgemv(cublas_handle_,
                    CUBLAS_OP_T,
                    n_,
                    i+1,
                    &one_,
                    d_V_,
                    n_,
                    &d_V_[(i+1)*n_],
                    1,
                    &zero_,
                    d_H_col_,
                    1);
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol

        cublasDgemv(cublas_handle_,
                    CUBLAS_OP_N,
                    n_,
                    i+1,
                    &minusone_,
                    d_V_,
                    n_,
                    d_H_col_,
                    1,
                    &one_,
                    &d_V_[n_*(i+1)],
                    1);
        // copy H_col to H

        cudaMemcpy(&h_H_[i*(restart_+1)], d_H_col_, sizeof(double) * (i+1), cudaMemcpyDeviceToHost);
        // add both pieces together (unstable otherwise, careful here!!)
        for(int j=0; j<=i; ++j) {
          h_H_[i*(restart_+1)+j] += h_aux_[j]; 
        }
        t = 0.0;
        cublasDdot (cublas_handle_,  n_, &d_V_[(i+1)*n_], 1, &d_V_[(i+1)*n_], 1, &t);

        //set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i)*(restart_+1)+i+1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1); 
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
        }
        break;
        // the two low synch schemes
      case 2:
        // KS: the kernels are limited by the size of the shared memory on the GPU. If too many vectors in Krylov space, use standard cublas routines.
        // V[1:i]^T[V[i] w]
        if(i < 200) {
          mass_inner_product_two_vectors(n_, i, &d_V_[i * n_],&d_V_[(i+1) * n_], d_V_, d_rvGPU_);
        } else {
          cublasDgemm(cublas_handle_,
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      i + 1,//m
                      2,//n
                      n_,//k
                      &one,//alpha
                      d_V_,//A
                      n_,//lda
                      &d_V_[i * n_],//B
                      n_,//ldb
                      &zero,
                      d_rvGPU_,//c
                      i+1);//ldc 
        } 
        // copy rvGPU to L
        cudaMemcpy(&h_L_[(i) * (restart_ + 1)], 
                   d_rvGPU_, 
                   (i + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(h_rv_, 
                   &d_rvGPU_[i + 1], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyDeviceToHost);

        for(int j=0; j<=i; ++j) {
          h_H_[(i)*(restart_+1)+j] = 0.0;
        }
        // triangular solve
        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
          } // for k
          h_H_[(i) * (restart_ + 1) + j] -= s; 
        }   // for j

        cudaMemcpy(d_Hcolumn_, 
                   &h_H_[(i) * (restart_ + 1)], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyHostToDevice);
        //again, use std cublas functions if Krylov space is too large
        if(i < 200) {
          mass_axpy(n_, i, d_V_, &d_V_[(i+1) * n_],d_Hcolumn_);
        } else {
          cublasDgemm(cublas_handle_,
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      n_,//m
                      1,//n
                      i + 1,//k
                      &minusone,//alpha
                      d_V_,//A
                      n_,//lda
                      d_Hcolumn_,//B
                      i + 1,//ldb
                      &one,
                      &d_V_[(i + 1) * n_],//c
                      n_);//ldc     

        }
        // normalize (second synch)
        t=0.0;
        cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

        // set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i) * (restart_ + 1) + i + 1] = t;
        if(t != 0.0) {
          t = 1.0/t;
          cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
        }
        break;

      case 3: //two synch Gauss-Seidel mgs, SUPER STABLE
        // according to unpublisjed work by ST
        // L is where we keep the triangular matrix(L is ON THE CPU)
        // if Krylov space is too large, use std cublas (because out of shared mmory)
        if(i < 200) {
          mass_inner_product_two_vectors(n_, i, &d_V_[i * n_],&d_V_[(i+1) * n_], d_V_, d_rvGPU_);
        } else {
          cublasDgemm(cublas_handle_,
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      i + 1,//m
                      2,//n
                      n_,//k
                      &one,//alpha
                      d_V_,//A
                      n_,//lda
                      &d_V_[i * n_],//B
                      n_,//ldb
                      &zero,
                      d_rvGPU_,//c
                      i+1);//ldc 
        }
        // copy rvGPU to L
        cudaMemcpy(&h_L_[(i) * (restart_ + 1)], 
                   d_rvGPU_, 
                   (i + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(h_rv_, 
                   &d_rvGPU_[i + 1], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyDeviceToHost);

        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = 0.0;
        }
        //triangular solve
        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
          } // for k
          h_H_[(i) * (restart_ + 1) + j] -= s;
        }   // for j

        // now compute h_rv = L^T h_H
        double h;
        for(int j = 0; j <= i; ++j) {
          // go through COLUMN OF L
          h_rv_[j] = 0.0;
          for(int k = j + 1; k <= i; ++k) {
            h = h_L_[k * (restart_ + 1) + j];
            h_rv_[j] += h_H_[(i) * (restart_ + 1) + k] * h;
          }
        }

        // and do one more tri solve with L^T: h_aux = (I-L)^{-1}h_rv
        for(int j = 0; j <= i; ++j) {
          h_aux_[j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_aux_[k];
          } // for k
          h_aux_[j] -= s;
        }   // for j

        // and now subtract that from h_H
        for(int j=0; j<=i; ++j) {
          h_H_[(i)*(restart_+1)+j] -= h_aux_[j];
        }
        cudaMemcpy(d_Hcolumn_,
                   &h_H_[(i) * (restart_ + 1)], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyHostToDevice);
        // if Krylov space too large, use std cublas routines
        if(i < 200) {
          mass_axpy(n_, i, d_V_, &d_V_[(i+1) * n_],d_Hcolumn_);
        } else {
          cublasDgemm(cublas_handle_,
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      n_,//m
                      1,//n
                      i + 1,//k
                      &minusone,//alpha
                      d_V_,//A
                      n_,//lda
                      d_Hcolumn_,//B
                      i + 1,//ldb
                      &one,
                      &d_V_[(i + 1) * n_],//c
                      n_);//ldc     
        }
        // normalize (second synch)
        t=0.0;
        cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

        // set the last entry in Hessenberg matrix
        t = sqrt(t);
        h_H_[(i) * (restart_ + 1) + i + 1] = t;
        if (t != 0.0){
          t = 1.0/t;
          cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
        }
        break;

      default:
        assert(0 && "Iterative refinement failed, wrong orthogonalization.\n");
        break;
    } // switch
  } // GramSchmidt



  hiopLinSolverSymSparseCUSOLVERGPU::hiopLinSolverSymSparseCUSOLVERGPU(const int& n, 
                                                                       const int& nnz, 
                                                                       hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparseCUSOLVER(n, nnz, nlp), 
      rhs_host_{nullptr},
      M_host_{nullptr}
  {
    rhs_host_ = LinearAlgebraFactory::create_vector("default", n);
    M_host_ = LinearAlgebraFactory::create_matrix_sparse("default", n, n, nnz);
  }
  
  hiopLinSolverSymSparseCUSOLVERGPU::~hiopLinSolverSymSparseCUSOLVERGPU()
  {
    delete rhs_host_;
    delete M_host_;
  }

  int hiopLinSolverSymSparseCUSOLVERGPU::matrixChanged()
  {
    size_type nnz = M_->numberOfNonzeros();
    double* mval_dev = M_->M();
    double* mval_host= M_host_->M();

    index_type* irow_dev = M_->i_row();
    index_type* irow_host= M_host_->i_row();

    index_type* jcol_dev = M_->j_col();
    index_type* jcol_host= M_host_->j_col();

    if("device" == nlp_->options->GetString("mem_space")) {
      checkCudaErrors(cudaMemcpy(mval_host, mval_dev, sizeof(double) * nnz, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(irow_host, irow_dev, sizeof(index_type) * nnz, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(jcol_host, jcol_dev, sizeof(index_type) * nnz, cudaMemcpyDeviceToHost));      
    } else {
      checkCudaErrors(cudaMemcpy(mval_host, mval_dev, sizeof(double) * nnz, cudaMemcpyHostToHost));
      checkCudaErrors(cudaMemcpy(irow_host, irow_dev, sizeof(index_type) * nnz, cudaMemcpyHostToHost));
      checkCudaErrors(cudaMemcpy(jcol_host, jcol_dev, sizeof(index_type) * nnz, cudaMemcpyHostToHost));
    }
    
    hiopMatrixSparse* swap_ptr = M_;
    M_ = M_host_;
    M_host_ = swap_ptr;
    
    int vret = hiopLinSolverSymSparseCUSOLVER::matrixChanged();

    swap_ptr = M_;
    M_ = M_host_;
    M_host_ = swap_ptr;
    
    return vret;
  }
  
  bool hiopLinSolverSymSparseCUSOLVERGPU::solve(hiopVector& x)
  {
    double* mval_dev = x.local_data();
    double* mval_host= rhs_host_->local_data();
   
    if("device" == nlp_->options->GetString("mem_space")) {
      checkCudaErrors(cudaMemcpy(mval_host, mval_dev, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    } else {
      checkCudaErrors(cudaMemcpy(mval_host, mval_dev, sizeof(double) * n_, cudaMemcpyHostToHost));
    }

    hiopMatrixSparse* swap_ptr = M_;
    M_ = M_host_;
    M_host_ = swap_ptr;

    bool bret = hiopLinSolverSymSparseCUSOLVER::solve(*rhs_host_);

    swap_ptr = M_;
    M_ = M_host_;
    M_host_ = swap_ptr;

    if("device" == nlp_->options->GetString("mem_space")) {
      checkCudaErrors(cudaMemcpy(mval_dev, mval_host, sizeof(double) * n_, cudaMemcpyHostToDevice));
    } else {
      checkCudaErrors(cudaMemcpy(mval_dev, mval_host, sizeof(double) * n_, cudaMemcpyHostToHost));     
    }
 
    return bret;
  }

} // namespace hiop
