// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

/**
 * @file hiopKKTLinSysSparseNormalEqn.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 */

#include "hiopKKTLinSysSparseNormalEqn.hpp"
#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverSymSparseMA57.hpp"
#endif

#ifdef HIOP_USE_CUDA
#include "hiopLinSolverCholCuSparse.hpp"
#endif

#include "hiopMatrixSparseCSRSeq.hpp"

namespace hiop
{
 
hiopKKTLinSysSparseNormalEqn::hiopKKTLinSysSparseNormalEqn(hiopNlpFormulation* nlp)
  : hiopKKTLinSysNormalEquation(nlp),
    nlpSp_{nullptr},
    HessSp_{nullptr},
    Jac_cSp_{nullptr},
    Jac_dSp_{nullptr},
    rhs_{nullptr},
    Hess_diag_{nullptr},
    dual_reg_{nullptr},
    Hxd_inv_{nullptr},
    JacD_{nullptr},
    JacDt_{nullptr},
    DiagJt_{nullptr},
    JDiagJt_{nullptr},
    Diag_reg_{nullptr},
    M_normaleqn_{nullptr},
    write_linsys_counter_(-1),
    csr_writer_(nlp)
{
  nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
  assert(nlpSp_);
}

hiopKKTLinSysSparseNormalEqn::~hiopKKTLinSysSparseNormalEqn()
{
  delete rhs_;
  delete Hess_diag_;
  delete dual_reg_;
  delete Hxd_inv_;
  delete JacD_;
  delete JacDt_;
  delete DiagJt_;
  delete JDiagJt_;
  delete Diag_reg_;
  delete M_normaleqn_;
}

bool hiopKKTLinSysSparseNormalEqn::build_kkt_matrix(const hiopVector& delta_wx_in,
                                                    const hiopVector& delta_wd_in,
                                                    const hiopVector& delta_cc_in,
                                                    const hiopVector& delta_cd_in)
{
  // TODO add is_equal
//  assert(delta_cc_in == delta_cd_in);
//  auto delta_cc = delta_cc_in;
   
  HessSp_ = dynamic_cast<hiopMatrixSparse*>(Hess_);
  if(!HessSp_) { assert(false); return false; }

  Jac_cSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_c_);
  if(!Jac_cSp_) { assert(false); return false; }

  Jac_dSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_d_);
  if(!Jac_dSp_) { assert(false); return false; }

  nlp_->runStats.kkt.tmUpdateInit.start();

  /* TODO: here we assume Hess is diagonal!*/
  assert(HessSp_->is_diagonal());
  
  hiopMatrixSymSparseTriplet* Hess_triplet = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  const hiopMatrixSparseTriplet* Jac_c_triplet = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_c_);
  const hiopMatrixSparseTriplet* Jac_d_triplet = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  
  assert(HessSp_ && Jac_cSp_ && Jac_dSp_);
  if(nullptr==Jac_dSp_ || nullptr==HessSp_) {
    nlp_->runStats.kkt.tmUpdateInit.stop();
    //incorrect linear algebra objects were provided to this class
    return false;
  }
  
  size_type nx = HessSp_->n();
  size_type neq = Jac_cSp_->m();
  size_type nineq = Jac_dSp_->m();
  assert(nineq == Dd_->get_size());
  assert(nx == Dx_->get_size());

  /* TODO: here we assume Hess is diagonal!*/
  if(nullptr == Hess_diag_) {
    Hess_diag_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    assert(Hess_diag_);
  }
  Hess_triplet->extract_diagonal(*Hess_diag_); 
 
  //build the diagonal Hx = Dx + delta_wx + diag(Hess)
  if(nullptr == Hx_) {
    Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    assert(Hx_);
  }
  Hx_->copyFrom(*Dx_);
  Hx_->axpy(1., delta_wx_in);
  Hx_->axpy(1., *Hess_diag_);
  
  // HD = Dd_ + delta_wd
  if(nullptr == Hd_) {
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nineq);
  }
  Hd_->copyFrom(*Dd_);
  // TODO: add function add_constant_with_bias()
  Hd_->axpy(1., delta_wd_in);

  nlp_->runStats.kkt.tmUpdateInit.stop();
  nlp_->runStats.kkt.tmUpdateLinsys.start();
  
  /*
  *  compute condensed linear system
  * ( [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1} [ Jc^T  Jd^T ] + [ delta_cc     0     ] ) 
  * ( [ Jd -I ] [   0           Dd+delta_wd ]      [  0     -I  ]   [    0      delta_cd ] ) 
  */

  // TODO: jump to the steps where we add dual regularization, if delta_wx is not changed and this function is called due to refactorization
  hiopTimer t;

  if(nullptr == JDiagJt_) {
    //first time this is called
    // form sparse matrix in triplet form
    size_type nnz_jac_con = Jac_cSp_->numberOfNonzeros()+Jac_dSp_->numberOfNonzeros()+nineq;
    auto* Jac_triplet_tmp = new hiopMatrixSparseTriplet(neq+nineq, nx+nineq, nnz_jac_con);
    Jac_triplet_tmp->setToZero();

    // build  [ Jc  0 ]
    //        [ Jd -I ]
    // copy Jac to the full iterate matrix
    size_type dest_nnz_st{0};
    Jac_triplet_tmp->copyRowsBlockFrom(*Jac_cSp_, 0,   neq,   0, dest_nnz_st);
    dest_nnz_st += Jac_cSp_->numberOfNonzeros();
    Jac_triplet_tmp->copyRowsBlockFrom(*Jac_dSp_, 0, nineq, neq, dest_nnz_st);
    dest_nnz_st += Jac_dSp_->numberOfNonzeros();

    // minus identity matrix for slack variables
    Jac_triplet_tmp->copyDiagMatrixToSubblock(-1., neq, nx, dest_nnz_st, nineq);
    dest_nnz_st += nineq;

    /// TODO: now we assume Jc and Jd won't change, i.e., LP or QP. hence we build JacD_ and JacDt_ once and save them
    Jac_triplet_tmp->sort();

    assert(nullptr == JacD_ && nullptr == JacDt_ && nullptr == DiagJt_);

    // symbolic conversion from triplet to CSR
    JacD_ = new hiopMatrixSparseCSRSeq();
    JacD_->form_from_symbolic(*Jac_triplet_tmp);
    JacD_->form_from_numeric(*Jac_triplet_tmp);

    JacDt_ = new hiopMatrixSparseCSRSeq();
    JacDt_->form_transpose_from_symbolic(*Jac_triplet_tmp);
    JacDt_->form_transpose_from_numeric(*Jac_triplet_tmp);

    DiagJt_ = new hiopMatrixSparseCSRSeq();
    DiagJt_->form_transpose_from_symbolic(*Jac_triplet_tmp);
    
    // build the diagonal Hxd_inv_ = [H+Dx+delta_wx, Dd+delta_wd ]^{-1}
    assert(nullptr == Hxd_inv_);
    Hxd_inv_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx + nineq);

    //symbolic multiplication for JacD*Diag*JacDt
    assert(nullptr == JDiagJt_);
    // J * (D*Jt)  (D is not used since it does not change the sparsity pattern)
    JDiagJt_ = JacD_->times_mat_alloc(*JacDt_);
    JacD_->times_mat_symbolic(*JDiagJt_, *JacDt_);
    
    delete Jac_triplet_tmp;
  }

  t.reset(); t.start();

  Hx_->copyToStarting(*Hxd_inv_, 0);
  Hd_->copyToStarting(*Hxd_inv_, nx);
  Hxd_inv_->invert();

  // J * D * Jt
  DiagJt_->copyFrom(*JacDt_);
  DiagJt_->scale_rows(*Hxd_inv_);
  JacD_->times_mat_numeric(0.0, *JDiagJt_, 1.0, *DiagJt_);

#ifdef HIOP_DEEPCHECKS
  JDiagJt_->check_csr_is_ordered();
#endif

  if(nullptr == M_normaleqn_) {
    //ensure storage for nonzeros diagonal is allocated by adding (symbolically) a diagonal matrix
    Diag_reg_ = new hiopMatrixSparseCSRSeq();
    
    // HD = Dd_ + delta_wd
    if(nullptr == dual_reg_) {
      dual_reg_ =LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), neq + nineq);
    }    
    Diag_reg_->form_diag_from_symbolic(*dual_reg_);

    //form sparsity pattern of M_condensed_ = JacD*Diag*JacDt + delta_cc*I
    M_normaleqn_ = Diag_reg_->add_matrix_alloc(*JDiagJt_);
    Diag_reg_->add_matrix_symbolic(*M_normaleqn_, *JDiagJt_);
  }

  t.reset(); t.start();
  
  Diag_reg_->setToZero();
  if(!delta_cc_in.is_zero()) {
    dual_reg_->startingAtCopyFromStartingAt(0, delta_cc_in, 0);
    dual_reg_->startingAtCopyFromStartingAt(neq, delta_cd_in, 0);
    
    Diag_reg_->addDiagonal(1.0,*dual_reg_);
  }
  Diag_reg_->add_matrix_numeric(*M_normaleqn_, 1.0, *JDiagJt_, 1.0);

  // TODO should have same code for different compute modes (remove is_cusolver_on), i.e., remove if(linSolver_ma57)
  // right now we use this if statement to transfer CSR form back to triplet form for ma57
  if(nullptr == linSys_) {
    linSys_ = determine_and_create_linsys(neq+nineq, M_normaleqn_->numberOfNonzeros());
  }

  {
    hiopLinSolverSymSparseMA57* linSolver_ma57 = dynamic_cast<hiopLinSolverSymSparseMA57*>(linSys_);
    if(linSolver_ma57) {
      auto* linSys = dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
      auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sys_matrix());
      assert(Msys);
      assert(Msys->m() == M_normaleqn_->m());

      index_type itnz = 0;
      for(index_type i = 0; i < Msys->m(); ++i) {
        for(index_type p = M_normaleqn_->i_row()[i]; p < M_normaleqn_->i_row()[i+1]; ++p) {
          const index_type j = M_normaleqn_->j_col()[p];
          if(i<=j) {
            Msys->i_row()[itnz] = i;
            Msys->j_col()[itnz] = j;
            Msys->M()[itnz] = M_normaleqn_->M()[p];
            itnz++; 
          }
        }
      }
      assert(itnz == Msys->numberOfNonzeros());
    }
  }

  t.stop();

  //write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_>=0) {
    // TODO csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nx, 0, nineq);
  }

  return true; 
}

hiopLinSolverSymSparse* hiopKKTLinSysSparseNormalEqn::determine_and_create_linsys(size_type n, size_type nnz)
{
  if(linSys_) {
    return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
  }

  if(nlp_->options->GetString("compute_mode") == "cpu") {
    //auto linear_solver = nlp_->options->GetString("linear_solver_sparse");

    //TODO:
    //add support for linear_solver == "cholmod"
    // maybe add pardiso as an option in the future
    //
    
#ifdef HIOP_USE_COINHSL
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_NormalEqn linsys: alloc MA57 for matrix of size %d (0 cons)\n", n);

    // only a triangular part is needed for ma57. reset nnz
    index_type itnz = 0;
    for(index_type i=0; i<JDiagJt_->m(); ++i) {
      for(index_type p=JDiagJt_->i_row()[i]; p<JDiagJt_->i_row()[i+1]; ++p) {
        const index_type j = JDiagJt_->j_col()[p];
        if(i<=j) {
          itnz++; 
        }
      }
    }
    linSys_ = new hiopLinSolverSymSparseMA57(n, itnz, nlp_);
#else
    assert(false && "HiOp was built without a sparse linear solver needed by the condensed KKT approach");
#endif // HIOP_USE_COINHSL
  } else {
    //
    // on device: compute_mode is hybrid, auto, or gpu
    //
    assert(nullptr==linSys_);
#ifdef HIOP_USE_CUDA
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_NormalEqn linsys: alloc cuSOLVER-chol matrix size %d\n", n);
    assert(JDiagJt_);
    linSys_ = new hiopLinSolverCholCuSparse(JDiagJt_, nlp_);
#endif
    //Return NULL (and assert) if a GPU sparse linear solver is not present
    assert(linSys_!=nullptr &&
           "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
           "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to 'cpu'");
  }
  assert(linSys_&& "KKT_SPARSE_NormalEqn linsys: cannot instantiate backend linear solver");
  return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
}

bool hiopKKTLinSysSparseNormalEqn::solveCompressed(hiopVector& ryc,
                                                   hiopVector& ryd,
                                                   hiopVector& dyc,
                                                   hiopVector& dyd)
{
  bool bret{false};

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  size_type nyc = ryc.get_size();
  size_type nyd = ryd.get_size();

  // this is rhs used by the direct "condensed" solve
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nyc+nyd);
  }

  nlp_->log->write("RHS KKT_SPARSE_NormalEqn ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_NormalEqn ryd:", ryd, hovIteration);

  //
  // form the rhs for the sparse linSys
  //
  ryc.copyToStarting(*rhs_, 0);
  ryd.copyToStarting(*rhs_, nyc);

  if(write_linsys_counter_>=0) {
    csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);
  }
  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  //
  // solve
  //
  nlp_->runStats.kkt.tmSolveInner.start();
  bret = linSys_->solve(*rhs_);
  nlp_->runStats.kkt.tmSolveInner.stop();

  nlp_->runStats.linsolv.end_linsolve();
 
  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "(summary for linear solver from KKT_SPARSE_NormalEqn(direct))\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }
  if(write_linsys_counter_>=0) {
    csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
  }

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  //
  // unpack
  //
  rhs_->startingAtCopyToStartingAt(0,   dyc, 0);
  rhs_->startingAtCopyToStartingAt(nyc, dyd, 0);
  nlp_->log->write("SOL KKT_SPARSE_NormalEqn dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_NormalEqn dyd:", dyd, hovMatrices);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();
    
  return bret;
}

int hiopKKTLinSysSparseNormalEqn::factorizeWithCurvCheck()
{
  //factorization
  size_type n_neg_eig = hiopKKTLinSysCurvCheck::factorizeWithCurvCheck();

  if(n_neg_eig == -1) {
    nlp_->log->printf(hovWarning,
                "KKT_SPARSE_NormalEqn linsys: Detected null eigenvalues.\n");
    n_neg_eig = -1;
  } else {
    // Cholesky factorization succeeds. Matrix is PD and hence the corresponding Augmented system has correct inertia
    n_neg_eig = Jac_c_->m() + Jac_d_->m();;    
  }

  return n_neg_eig;
}

} // end of namespace
