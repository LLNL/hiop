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
#include <IterativeRefinement.hpp>
#include <RefactorizationSolver.hpp>
#include <MatrixCsr.hpp>

#include "hiop_blasdefs.hpp"
#include "KrylovSolverKernels.h"

#include "cusparse_v2.h"
#include <sstream>
#include <string>

#define checkCudaErrors(val) hiopCheckCudaError((val), __FILE__, __LINE__)

namespace hiop
{
  hiopLinSolverSymSparseReSolve::hiopLinSolverSymSparseReSolve(const int& n, 
                                                               const int& nnz, 
                                                               hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp), 
      index_covert_CSR2Triplet_{ nullptr },
      index_covert_extra_Diag2CSR_{ nullptr }, 
      n_{ n },
      nnz_{ 0 },
      factorizationSetupSucc_{ 0 },
      is_first_call_{ true }
  {
    solver_ = new ReSolve::RefactorizationSolver(n);
    rhs_    = new double[n]{ 0 };
    // std::cout << "n: " << solver_->n_ << ", nnz: " << solver_->nnz_ << "\n";


    // Select matrix ordering
    int ordering = 1;
    std::string ord = nlp_->options->GetString("linear_solver_sparse_ordering");
    if(ord == "amd_ssparse") {
      ordering = 0;
    } else if(ord == "colamd_ssparse") {
      ordering = 1;
    } else {
      nlp_->log->printf(hovWarning, 
                        "Ordering %s not compatible with cuSOLVER LU, using default ...\n",
                        ord.c_str());
      ordering = 1;
    }
    solver_->ordering() = ordering;
    std::cout << "Ordering: " << solver_->ordering() << "\n";

    // Select factorization
    std::string fact;
    fact = nlp_->options->GetString("resolve_factorization");
    if(fact != "klu") {
      nlp_->log->printf(hovWarning,
                        "Factorization %s not compatible with cuSOLVER LU, using default ...\n",
                        fact.c_str());
      fact = "klu";
    }
    solver_->fact() = fact;
    std::cout << "Factorization: " << solver_->fact() << "\n";

    // Select refactorization
    std::string refact;
    refact = nlp_->options->GetString("resolve_refactorization");
    if(refact != "glu" && refact != "rf") {
      nlp_->log->printf(hovWarning, 
                        "Refactorization %s not compatible with cuSOLVER LU, using default ...\n",
                        refact.c_str());
      refact = "glu";
    }
    solver_->refact() = refact;
    std::cout << "Refactorization: " << solver_->refact() << "\n";

    // by default, dont use iterative refinement
    std::string use_ir;
    int maxit_test  = nlp_->options->GetInteger("ir_inner_maxit");

    if ((maxit_test < 0) || (maxit_test > 1000)){
      nlp_->log->printf(hovWarning, 
                        "Wrong maxit value: %d. Use int maxit value between 0 and 1000. Setting default (50)  ...\n",
                        maxit_test);
      maxit_test = 50;
    }
    use_ir = "no";
    if(maxit_test > 0){
      use_ir = "yes";
      solver_->enable_iterative_refinement();
      solver_->ir()->maxit() = maxit_test;
    } 
    if(use_ir == "yes") {
      if((refact == "rf")) {

        solver_->ir()->restart() =  nlp_->options->GetInteger("ir_inner_restart");

        if ((solver_->ir()->restart() <0) || (solver_->ir()->restart() >100)){
          nlp_->log->printf(hovWarning, 
                            "Wrong restart value: %d. Use int restart value between 1 and 100. Setting default (20)  ...\n",
                            solver_->ir()->restart());
          solver_->ir()->restart() = 20;
        }


        solver_->ir()->tol()  = nlp_->options->GetNumeric("ir_inner_tol");
        if ((solver_->ir()->tol() <0) || (solver_->ir()->tol() >1)){
          nlp_->log->printf(hovWarning, 
                            "Wrong tol value: %e. Use double tol value between 0 and 1. Setting default (1e-12)  ...\n",
                            solver_->ir()->tol());
          solver_->ir()->tol() = 1e-12;
        }
        solver_->ir()->orth_option() = nlp_->options->GetString("ir_inner_cusolver_gs_scheme");

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
        if(solver_->ir()->orth_option() != "mgs" && solver_->ir()->orth_option() != "cgs2" && solver_->ir()->orth_option() != "mgs_two_synch" && solver_->ir()->orth_option() != "mgs_pm") {
          nlp_->log->printf(hovWarning, 
                            "mgs option : %s is wrong. Use 'mgs', 'cgs2', 'mgs_two_synch' or 'mgs_pm'. Switching to default (mgs) ...\n",
                            use_ir.c_str());
          solver_->ir()->orth_option() = "mgs";
        }

        solver_->ir()->conv_cond() =  nlp_->options->GetInteger("ir_inner_conv_cond");

        if ((solver_->ir()->conv_cond() <0) || (solver_->ir()->conv_cond() >2)){
          nlp_->log->printf(hovWarning, 
                            "Wrong IR convergence condition: %d. Use int value: 0, 1 or 2. Setting default (0)  ...\n",
                            solver_->ir()->conv_cond());
          solver_->ir()->conv_cond() = 0;
        }

      } else {
        nlp_->log->printf(hovWarning, 
                          "Currently, inner iterative refinement works ONLY with cuSolverRf ... \n");
        use_ir = "no";
      }
    }
    solver_->use_ir() = use_ir;
    std::cout << "Use IR: " << solver_->use_ir() << "\n";
  } // constructor

  hiopLinSolverSymSparseReSolve::~hiopLinSolverSymSparseReSolve()
  {
    delete solver_;
    delete [] rhs_;

    // Delete CSR <--> triplet mappings
    delete[] index_covert_CSR2Triplet_;
    delete[] index_covert_extra_Diag2CSR_;
  }

  int hiopLinSolverSymSparseReSolve::matrixChanged()
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
      int retval = solver_->factorize();
      if(retval == -1) {
        nlp_->log->printf(hovWarning, "Numeric klu factorization failed. Regularizing ...\n");
        // This is not a catastrophic failure
        // The matrix is singular so return -1 to regularaize!
        return -1;
      } else { // Numeric was succesfull so now can set up
        solver_->setup_refactorization();
        factorizationSetupSucc_ = 1;
        nlp_->log->printf(hovScalars, "Numeric klu factorization succesful! \n");
      }
    } else { // factorizationSetupSucc_ == 1
      solver_->refactorize();
    }

    nlp_->runStats.linsolv.tmFactTime.stop();
    return 0;
  }

  bool hiopLinSolverSymSparseReSolve::solve(hiopVector& x)
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

    double* dx = x.local_data();
    memcpy(rhs_, dx, n_*sizeof(double));
    checkCudaErrors(cudaMemcpy(solver_->devr(), rhs_, sizeof(double) * n_, cudaMemcpyHostToDevice));

    bool retval = solver_->triangular_solve(dx, rhs_, ir_tol);
    if(!retval) {
      nlp_->log->printf(hovError,  // catastrophic failure
                        "ReSolve triangular solver failed\n");
    }

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return true;
  }

  void hiopLinSolverSymSparseReSolve::firstCall()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_ > 0);

    // Transfer triplet to CSR form

    // Allocate row pointers and compute number of nonzeros.
    solver_->mat_A_csr()->allocate_size(n_);
    compute_nnz();
    solver_->set_nnz(nnz_);

    // Allocate column indices and matrix values
    solver_->mat_A_csr()->allocate_nnz(nnz_);

    // Set column indices and matrix values.
    set_csr_indices_values();

    // Copy matrix to device
    solver_->mat_A_csr()->update_from_host_mirror();

    if(solver_->use_ir() == "yes") {
      solver_->setup_iterative_refinement_matrix(n_, nnz_);
    }
    /*
     * initialize matrix factorization
     */
    if(solver_->setup_factorization() < 0) {
      nlp_->log->printf(hovError,  // catastrophic failure
                        "Symbolic factorization failed!\n");
      return;
    };
    is_first_call_ = false;
  }

  void hiopLinSolverSymSparseReSolve::update_matrix_values()
  {
    double* vals = solver_->mat_A_csr()->get_vals_host();
    // update matrix
    for(int k = 0; k < nnz_; k++) {
      vals[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(int i = 0; i < n_; i++) {
      if(index_covert_extra_Diag2CSR_[i] != -1)
        vals[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
  }


  void hiopLinSolverSymSparseReSolve::compute_nnz()
  {
    //
    // compute nnz in each row
    //
    int* row_ptr = solver_->mat_A_csr()->get_irows_host();

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

  void hiopLinSolverSymSparseReSolve::set_csr_indices_values()
  {
    //
    // set correct col index and value
    //
    const int* row_ptr = solver_->mat_A_csr()->get_irows_host();
    int*       col_idx = solver_->mat_A_csr()->get_jcols_host();
    double*    vals    = solver_->mat_A_csr()->get_vals_host();

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

  // Error checking utility for CUDA
  // KS: might later become part of src/Utils, putting it here for now
  template <typename T>
  void hiopLinSolverSymSparseReSolve::hiopCheckCudaError(T result,
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



  hiopLinSolverSymSparseReSolveGPU::hiopLinSolverSymSparseReSolveGPU(const int& n, 
                                                                     const int& nnz, 
                                                                     hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparseReSolve(n, nnz, nlp), 
      rhs_host_{nullptr},
      M_host_{nullptr}
  {
    rhs_host_ = LinearAlgebraFactory::create_vector("default", n);
    M_host_ = LinearAlgebraFactory::create_matrix_sparse("default", n, n, nnz);
  }
  
  hiopLinSolverSymSparseReSolveGPU::~hiopLinSolverSymSparseReSolveGPU()
  {
    delete rhs_host_;
    delete M_host_;
  }

  int hiopLinSolverSymSparseReSolveGPU::matrixChanged()
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
    
    int vret = hiopLinSolverSymSparseReSolve::matrixChanged();

    swap_ptr = M_;
    M_ = M_host_;
    M_host_ = swap_ptr;
    
    return vret;
  }
  
  bool hiopLinSolverSymSparseReSolveGPU::solve(hiopVector& x)
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

    bool bret = hiopLinSolverSymSparseReSolve::solve(*rhs_host_);

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

