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
#include "IterativeRefinement.hpp"

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
    mat_A_csr_ = new ReSolve::MatrixCsr();

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
                        maxit_test);
      maxit_test = 50;
    }
    use_ir_ = "no";
    if(maxit_test > 0){
      use_ir_ = "yes";
      ir_ = new ReSolve::IterativeRefinement();
      ir_->maxit() = maxit_test;
    } 
    if(use_ir_ == "yes") {
      if((refact_ == "rf")) {

        ir_->restart() =  nlp_->options->GetInteger("ir_inner_restart");

        if ((ir_->restart() <0) || (ir_->restart() >100)){
          nlp_->log->printf(hovWarning, 
                            "Wrong restart value: %d. Use int restart value between 1 and 100. Setting default (20)  ...\n",
                            ir_->restart());
          ir_->restart() = 20;
        }


        ir_->tol()  = nlp_->options->GetNumeric("ir_inner_tol");
        if ((ir_->tol() <0) || (ir_->tol() >1)){
          nlp_->log->printf(hovWarning, 
                            "Wrong tol value: %e. Use double tol value between 0 and 1. Setting default (1e-12)  ...\n",
                            ir_->tol());
          ir_->tol() = 1e-12;
        }
        ir_->orth_option() = nlp_->options->GetString("ir_inner_cusolver_gs_scheme");

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
         * that the vectors are orthogonal to the machine epsilon, as in cgs2. Since Stephen's paper is named "post modern GMRES", we call this Gram-Schmidt scheme "mgs_pm".
         */ 
        if(ir_->orth_option() != "mgs" && ir_->orth_option() != "cgs2" && ir_->orth_option() != "mgs_two_synch" && ir_->orth_option() != "mgs_pm") {
          nlp_->log->printf(hovWarning, 
                            "mgs option : %s is wrong. Use 'mgs', 'cgs2', 'mgs_two_synch' or 'mgs_pm'. Switching to default (mgs) ...\n",
                            use_ir_.c_str());
          ir_->orth_option() = "mgs";
        }

        ir_->conv_cond() =  nlp_->options->GetInteger("ir_inner_conv_cond");

        if ((ir_->conv_cond() <0) || (ir_->conv_cond() >2)){
          nlp_->log->printf(hovWarning, 
                            "Wrong IR convergence condition: %d. Use int value: 0, 1 or 2. Setting default (0)  ...\n",
                            ir_->conv_cond());
          ir_->conv_cond() = 0;
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
    delete mat_A_csr_;

    // Delete CSR <--> triplet mappings
    delete[] index_covert_CSR2Triplet_;
    delete[] index_covert_extra_Diag2CSR_;

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

    if(is_first_call_) {
      firstCall();
    } else {
      update_matrix_values();
    } 

    if(factorizationSetupSucc_ == 0) {
      int retval = factorize();
      if(retval == -1) {
        nlp_->log->printf(hovWarning, "Numeric klu factorization failed. Regularizing ...\n");
        // This is not a catastrophic failure
        // The matrix is singular so return -1 to regularaize!
        return -1;
      } else { // Numeric was succesfull so now can set up
        setup_refactorization();
        factorizationSetupSucc_ = 1;
        nlp_->log->printf(hovScalars, "Numeric klu factorization succesful! \n");
      }
    } else { // factorizationSetupSucc_ == 1
      refactorize();
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

    hiopVector* rhs = x.new_copy(); // TODO: Allocate this as a part of the solver workspace!
    double* dx = x.local_data();
    double* drhs = rhs->local_data();
    checkCudaErrors(cudaMemcpy(devr_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    bool retval = triangular_solve(dx, drhs, ir_tol);

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
    // kRowPtr_ = new int[n_ + 1]{ 0 };
    mat_A_csr_->allocate_size(n_);
    compute_nnz();

    // Allocate column indices and matrix values
    // kVal_ = new double[nnz_]{ 0.0 };
    // jCol_ = new int[nnz_]{ 0 };
    mat_A_csr_->allocate_nnz(nnz_);
    // Set column indices and matrix values.
    set_csr_indices_values();

    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));

    // Copy matrix to device
    mat_A_csr_->update_from_host_mirror();

    if(use_ir_ == "yes") {
      // ir_->setup_system_matrix(n_, nnz_, dia_, dja_, da_);
      ir_->setup_system_matrix(n_, nnz_, mat_A_csr_->get_irows(), mat_A_csr_->get_jcols(), mat_A_csr_->get_vals());
    }
    /*
     * initialize matrix factorization
     */
    setup_factorization();
    is_first_call_ = false;
  }

  void hiopLinSolverSymSparseCUSOLVER::update_matrix_values()
  {
    double* vals = mat_A_csr_->get_vals_host();
    // update matrix
    for(int k = 0; k < nnz_; k++) {
      vals[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(int i = 0; i < n_; i++) {
      if(index_covert_extra_Diag2CSR_[i] != -1)
        vals[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
    // std::cout << "Updated matrix values ...\n";
  }

  int hiopLinSolverSymSparseCUSOLVER::setup_factorization()
  {
    int* row_ptr = mat_A_csr_->get_irows_host();
    int* col_idx = mat_A_csr_->get_jcols_host();

    if(fact_ == "klu") {
      /* initialize KLU setup parameters, dont factorize yet */
      initializeKLU();

      /*perform KLU but only the symbolic analysis (important)   */

      klu_free_symbolic(&Symbolic_, &Common_);
      klu_free_numeric(&Numeric_, &Common_);
      Symbolic_ = klu_analyze(n_, row_ptr, col_idx, &Common_);

      if(Symbolic_ == nullptr) {
        nlp_->log->printf(hovError,  // catastrophic failure
                          "Symbolic factorization failed!\n");
      }
    } else { // for future
      assert(0 && "Only KLU is available for the first factorization.\n");
    }
    return 0;
  }

  int hiopLinSolverSymSparseCUSOLVER::factorize()
  {
    // Numeric_ = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic_, &Common_);
    Numeric_ = klu_factor(mat_A_csr_->get_irows_host(), mat_A_csr_->get_jcols_host(), mat_A_csr_->get_vals_host(), Symbolic_, &Common_);
    return (Numeric_ == nullptr) ? -1 : 0;
  }

  void hiopLinSolverSymSparseCUSOLVER::setup_refactorization()
  {
    if(refact_ == "glu") {
      initializeCusolverGLU();
      refactorizationSetupCusolverGLU();
    } else if(refact_ == "rf") {
      initializeCusolverRf();
      refactorizationSetupCusolverRf();
      if(use_ir_ == "yes") {
        IRsetup();
        // std::cout << "IR is set up successfully ...\n";
      }
    } else { // for future -
      assert(0 && "Only glu and rf refactorizations available.\n");
    }
  }

  int hiopLinSolverSymSparseCUSOLVER::refactorize()
  {
    // checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat_A_csr_->get_vals(), mat_A_csr_->get_vals_host(), sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    // re-factor here

    if(refact_ == "glu") {
      sp_status_ = cusolverSpDgluReset(handle_cusolver_, 
                                        n_,
                                        /* A is original matrix */
                                        nnz_,
                                        descr_A_,
                                        mat_A_csr_->get_vals(),  //da_,
                                        mat_A_csr_->get_irows(), //dia_,
                                        mat_A_csr_->get_jcols(), //dja_,
                                        info_M_);
      sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
    } else {
      if(refact_ == "rf") {
        sp_status_ = cusolverRfResetValues(n_, 
                                            nnz_, 
                                            mat_A_csr_->get_irows(), //dia_,
                                            mat_A_csr_->get_jcols(), //dja_,
                                            mat_A_csr_->get_vals(),  //da_,
                                            d_P_,
                                            d_Q_,
                                            handle_rf_);
        cudaDeviceSynchronize();
        sp_status_ = cusolverRfRefactor(handle_rf_);
      }
    }
    // end of factor
    return 0;
  }

  bool hiopLinSolverSymSparseCUSOLVER::triangular_solve(double* dx, const double* drhs, double tol)
  {
    // solve HERE
    if(refact_ == "glu") {
      sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                       n_,
                                       /* A is original matrix */
                                       nnz_,
                                       descr_A_,
                                       mat_A_csr_->get_vals(),  //da_,
                                       mat_A_csr_->get_irows(), //dia_,
                                       mat_A_csr_->get_jcols(), //dja_,
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
                          "GLU solve failed with status: %d\n", 
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
              ir_->set_tol(tol);
              nlp_->log->printf(hovScalars,
                                "Running iterative refinement with tol %e\n", tol);
              checkCudaErrors(cudaMemcpy(devx_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

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
                              "Rf solve failed with status: %d\n", 
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
    return true;
  }

  void hiopLinSolverSymSparseCUSOLVER::compute_nnz()
  {
    //
    // compute nnz in each row
    //
    int* row_ptr = mat_A_csr_->get_irows_host();

    // off-diagonal part
    row_ptr[0] = 0;
    for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
      if(M_->i_row()[k] != M_->j_col()[k]) {
        row_ptr[M_->i_row()[k] + 1]++;
        row_ptr[M_->j_col()[k] + 1]++;
        nnz_ += 2;
      }
    }
    // diagonal part
    for(int i = 0; i < n_; i++) {
      row_ptr[i + 1]++;
      nnz_ += 1;
    }
    // get correct row ptr index
    for(int i = 1; i < n_ + 1; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }
    assert(nnz_ == row_ptr[n_]);
  }

  void hiopLinSolverSymSparseCUSOLVER::set_csr_indices_values()
  {
    //
    // set correct col index and value
    //
    const int* row_ptr = mat_A_csr_->get_irows_host();
    int*       col_idx = mat_A_csr_->get_jcols_host();
    double*    vals    = mat_A_csr_->get_vals_host();

    index_covert_CSR2Triplet_    = new int[nnz_];
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
        nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];
        col_idx[nnz_tmp] = colID_tmp;
        vals[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        vals[nnz_tmp] += M_->M()[M_->numberOfNonzeros() - n_ + rowID_tmp];
        index_covert_extra_Diag2CSR_[rowID_tmp] = nnz_tmp;

        nnz_each_row_tmp[rowID_tmp]++;
        total_nnz_tmp++;
      } else {
        nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];
        col_idx[nnz_tmp] = colID_tmp;
        vals[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        nnz_tmp = nnz_each_row_tmp[colID_tmp] + row_ptr[colID_tmp];
        col_idx[nnz_tmp] = rowID_tmp;
        vals[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        nnz_each_row_tmp[rowID_tmp]++;
        nnz_each_row_tmp[colID_tmp]++;
        total_nnz_tmp += 2;
      }
    }
    // correct the missing dia_gonal term
    for(int i = 0; i < n_; i++) {
      if(nnz_each_row_tmp[i] != row_ptr[i + 1] - row_ptr[i]) {
        assert(nnz_each_row_tmp[i] == row_ptr[i + 1] - row_ptr[i] - 1);
        nnz_tmp = nnz_each_row_tmp[i] + row_ptr[i];
        col_idx[nnz_tmp] = i;
        vals[nnz_tmp] = M_->M()[M_->numberOfNonzeros() - n_ + i];
        index_covert_CSR2Triplet_[nnz_tmp] = M_->numberOfNonzeros() - n_ + i;
        total_nnz_tmp += 1;

        std::vector<int> ind_temp(row_ptr[i + 1] - row_ptr[i]);
        std::iota(ind_temp.begin(), ind_temp.end(), 0);
        std::sort(ind_temp.begin(), ind_temp.end(), 
                  [&](int a, int b) {
                  return col_idx[a + row_ptr[i]] < col_idx[b + row_ptr[i]];
                  }
                 );

        reorder(vals + row_ptr[i], ind_temp, row_ptr[i + 1] - row_ptr[i]);
        reorder(index_covert_CSR2Triplet_ + row_ptr[i], ind_temp, row_ptr[i + 1] - row_ptr[i]);
        std::sort(col_idx + row_ptr[i], col_idx + row_ptr[i + 1]);
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
                                     mat_A_csr_->get_irows_host(), //kRowPtr_,
                                     mat_A_csr_->get_jcols_host(), //jCol_, 
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
                                     mat_A_csr_->get_vals(),  //da_, 
                                     mat_A_csr_->get_irows(), //dia_, 
                                     mat_A_csr_->get_jcols(), //dja_, 
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
                                       mat_A_csr_->get_irows(), //dia_,
                                       mat_A_csr_->get_jcols(), //dja_,
                                       mat_A_csr_->get_vals(),  //da_,
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
    ir_->setup(handle_, handle_cublas_, handle_rf_, n_, d_T_, d_P_, d_Q_, devx_, devr_);
  }
  // Experimental code ends here


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

namespace ReSolve {
  MatrixCsr::MatrixCsr()
  {
  }

  MatrixCsr::~MatrixCsr()
  {
    if(n_ == 0)
      return;

    cudaFree(irows_);
    cudaFree(jcols_);
    cudaFree(vals_);

    delete [] irows_host_;
    delete [] jcols_host_;
    delete [] vals_host_ ;
  }

  void MatrixCsr::allocate_size(int n)
  {
    n_ = n;
    (cudaMalloc(&irows_, (n_+1) * sizeof(int)));
    irows_host_ = new int[n_+1]{0};
  }

  void MatrixCsr::allocate_nnz(int nnz)
  {
    nnz_ = nnz;
    (cudaMalloc(&jcols_, nnz_ * sizeof(int)));
    (cudaMalloc(&vals_,  nnz_ * sizeof(double)));
    jcols_host_ = new int[nnz_]{0};
    vals_host_  = new double[nnz_]{0};
  }

  void MatrixCsr::update_from_host_mirror()
  {
    (cudaMemcpy(irows_, irows_host_, sizeof(int)    * (n_+1), cudaMemcpyHostToDevice));
    (cudaMemcpy(jcols_, jcols_host_, sizeof(int)    * nnz_,   cudaMemcpyHostToDevice));
    (cudaMemcpy(vals_,  vals_host_,  sizeof(double) * nnz_,   cudaMemcpyHostToDevice));
  }

  void MatrixCsr::copy_to_host_mirror()
  {
    (cudaMemcpy(irows_host_, irows_, sizeof(int)    * (n_+1), cudaMemcpyDeviceToHost));
    (cudaMemcpy(jcols_host_, jcols_, sizeof(int)    * nnz_,   cudaMemcpyDeviceToHost));
    (cudaMemcpy(vals_host_,  vals_,  sizeof(double) * nnz_,   cudaMemcpyDeviceToHost));
  }
}
