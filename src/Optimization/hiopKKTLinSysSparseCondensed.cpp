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
 * @file hiopKKTLinSysSparseCondensed.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 */

#include "hiopKKTLinSysSparseCondensed.hpp"

#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverSymSparseMA57.hpp"
#endif

#ifdef HIOP_USE_CUDA
#include "hiopLinSolverCholCuSparse.hpp"
#include "hiopMatrixSparseCSRCUDA.hpp"
#endif

#include "hiopMatrixSparseTripletStorage.hpp"
#include "hiopMatrixSparseCSRSeq.hpp"

namespace hiop
{

 
hiopKKTLinSysCondensedSparse::hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressedSparseXDYcYd(nlp),
    JacD_(nullptr),
    JacDt_(nullptr),
    JtDiagJ_(nullptr),
    Hess_lower_csr_(nullptr),
    Hess_upper_csr_(nullptr),
    Hess_csr_(nullptr),
    M_condensed_(nullptr),
    delta_wx_(0.)
{
}

hiopKKTLinSysCondensedSparse::~hiopKKTLinSysCondensedSparse()
{
  delete M_condensed_;
  delete JtDiagJ_;
  delete JacDt_;
  delete JacD_;
  delete Hess_csr_;
  delete Hess_upper_csr_;
  delete Hess_lower_csr_;
}

bool hiopKKTLinSysCondensedSparse::build_kkt_matrix(const double& delta_wx_in,
                                                    const double& delta_wd_in,
                                                    const double& dcc,
                                                    const double& dcd)
{
  nlp_->runStats.kkt.tmUpdateInit.start();
  
  auto delta_wx = delta_wx_in;
  auto delta_wd = delta_wd_in;
  if(dcc!=0) {
    //nlp_->log->printf(hovWarning, "LinSysCondensed: dual reg. %.6e primal %.6e %.6e\n", dcc, delta_wx, delta_wd);
    assert(dcc == dcd);
    delta_wx += fabs(dcc);
    delta_wd += fabs(dcc);
  }
  delta_wx_ = delta_wx;

  hiopMatrixSymSparseTriplet* Hess_triplet = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  HessSp_ = Hess_triplet; //dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  
  Jac_cSp_ = nullptr; //not used by this class

  const hiopMatrixSparseTriplet* Jac_triplet = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  Jac_dSp_ = Jac_triplet;
  
  assert(HessSp_ && Jac_dSp_);
  if(nullptr==Jac_dSp_ || nullptr==HessSp_) {
    nlp_->runStats.kkt.tmUpdateInit.stop();
    //incorrect linear algebra objects were provided to this class
    return false;
  }

  assert(0 == Jac_c_->m() &&
         "Detected NLP with equality constraints. Please use hiopNlpSparseIneq formulation");
  
  size_type nx = HessSp_->n();
  size_type nineq = Jac_dSp_->m();
  assert(nineq == Dd_->get_size());
  assert(nx == Dx_->get_size());

  if(nullptr == Hd_) {
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nineq);
  }
  Hd_->copyFrom(*Dd_);  
  Hd_->addConstant(delta_wd);

  nlp_->runStats.kkt.tmUpdateInit.stop();
  nlp_->runStats.kkt.tmUpdateLinsys.start();
  
  //
  // compute condensed linear system J'*D*J + H + Dx + delta_wx*I
  //

//#define CSRCUDA_TESTING 1
  
  hiopTimer t;

  hiopMatrixSparseCSRCUDA JacD_cuda;
  hiopMatrixSparseCSRCUDA JacDt_cuda;
  
  // symbolic conversion from triplet to CSR
  if(nullptr == JacD_) {
    t.reset(); t.start();
    JacD_ = new hiopMatrixSparseCSRSeq();
    JacD_->form_from_symbolic(*Jac_triplet);

    assert(nullptr == JacDt_);
    JacDt_ = new hiopMatrixSparseCSRSeq();
    JacDt_->form_transpose_from_symbolic(*JacD_);
    //t.stop(); printf("JacD JacDt-symb from csr    took %.5f\n", t.getElapsedTime());
  }

  // numeric conversion from triplet to CSR
  t.reset(); t.start();
  JacD_->form_from_numeric(*Jac_triplet);
  JacDt_->form_transpose_from_numeric(*JacD_);
  //t.stop(); printf("JacD JacDt-nume csr    took %.5f\n", t.getElapsedTime());

#ifdef CSRCUDA_TESTING
  JacD_cuda.form_from_symbolic(*Jac_triplet);
  JacD_cuda.form_from_numeric(*Jac_triplet);
  JacD_cuda.print();
  JacDt_cuda.form_transpose_from_symbolic(JacD_cuda);
  JacDt_cuda.form_transpose_from_numeric(JacD_cuda);
  JacDt_cuda.print();
#endif
  
  //symbolic multiplication for JacD'*D*J
  if(nullptr == JtDiagJ_) {
    t.reset(); t.start();
    
    // D * J
    //nothing to do symbolically since we just numerically scale columns of Jt by D 
  
    // Jt* (D*J)  (D is not used since it does not change the sparsity pattern)
    JtDiagJ_ = JacDt_->times_mat_alloc(*JacD_);
    JacDt_->times_mat_symbolic(*JtDiagJ_, *JacD_);
    //t.stop(); printf("J*D*J'-symb  took %.5f\n", t.getElapsedTime());
  }
  
  //numeric multiplication for JacD'*D*J
  t.reset(); t.start();
  // Jt * D
  JacD_->scale_rows(*Hd_);
  // (Jt*D) * J
  JacDt_->times_mat_numeric(0.0, *JtDiagJ_, 1.0, *JacD_);
  //t.stop(); printf("J*D*J'-nume  took %.5f\n", t.getElapsedTime());
  
#ifdef HIOP_DEEPCHECKS
  JtDiagJ_->check_csr_is_ordered();
#endif

  if(nullptr == linSys_) {
    //first time this is called
    assert(nullptr == Hess_upper_csr_);
    Hess_upper_csr_ = new hiopMatrixSparseCSRSeq();
    Hess_upper_csr_->form_from_symbolic(*Hess_triplet);

    assert(nullptr == Hess_lower_csr_);
    Hess_lower_csr_ = new hiopMatrixSparseCSRSeq();
    Hess_lower_csr_->form_transpose_from_symbolic(*Hess_triplet);

    assert(Hess_lower_csr_->numberOfNonzeros() == Hess_upper_csr_->numberOfNonzeros());

    assert(nullptr == Hess_csr_);
    Hess_csr_ = Hess_lower_csr_->add_matrix_alloc(*Hess_upper_csr_);
    Hess_lower_csr_->add_matrix_symbolic(*Hess_csr_, *Hess_upper_csr_);

    
    //a temporary matrix needed to form sparsity pattern of M_condensed_
    auto* M_condensed_tmp = Hess_csr_->add_matrix_alloc(*JtDiagJ_);
    Hess_csr_->add_matrix_symbolic(*M_condensed_tmp, *JtDiagJ_);
    
    //ensure storage for nonzeros diagonal is allocated by adding (symbolically)
    //a diagonal matrix
    hiopMatrixSparseCSRSeq Diag;
    Diag.form_diag_from_symbolic(*Dx_);

    assert(nullptr == M_condensed_);
    M_condensed_ = M_condensed_tmp->add_matrix_alloc(Diag);
    M_condensed_tmp->add_matrix_symbolic(*M_condensed_, Diag);
    delete M_condensed_tmp;

    //t.stop(); printf("ADD-symb  took %.5f\n", t.getElapsedTime());
  } else {
    auto* lins_sys_sparse = dynamic_cast<hiopLinSolverSymSparse*>(linSys_);
    assert(linSys_);
    assert(M_condensed_);
    //todo assert(M_condensed_ == linSys_->sys_matrix());
  }

  t.reset(); t.start();
  Hess_upper_csr_->form_from_numeric(*Hess_triplet);
  Hess_upper_csr_->set_diagonal(0.0);
  Hess_lower_csr_->form_transpose_from_numeric(*Hess_triplet);
  //
  // Hess_csr_ = Hess_lower_csr_ + Hess_upper_csr_
  //
  Hess_lower_csr_->add_matrix_numeric(*Hess_csr_, 1.0, *Hess_upper_csr_, 1.0);

  //
  // M_condensed_ = M_condensed_ + Hess_csr_ + JtDiagJ_ + Dx_ + delta_wx*I
  //
  Hess_csr_->add_matrix_numeric(*M_condensed_, 1.0, *JtDiagJ_, 1.0);

  if(delta_wx>0) {
    M_condensed_->addDiagonal(delta_wx);
  }

  M_condensed_->addDiagonal(1.0, *Dx_);
  //t.stop(); printf("ADD-nume  took %.5f\n", t.getElapsedTime());
  
  int nnz_condensed = M_condensed_->numberOfNonzeros();

  //
  // linear system matrix update
  //

  // TODO should have same code for different compute modes (remove is_cusolver_on)
  
  bool is_cusolver_on = nlp_->options->GetString("compute_mode") == "cpu" ? false : true;
#ifndef HIOP_USE_CUDA
  if(is_cusolver_on) {
    nlp_->log->printf(hovWarning,
                      "hiopKKTLinSysCondensedSparse: HiOp was built without CUDA and will use a CPU "
                      "linear solver (MA57)\n");
    is_cusolver_on = false;
  }
#endif  
  if(is_cusolver_on) {
    linSys_ = determine_and_create_linsys(nx, nineq, M_condensed_->numberOfNonzeros());
  } else {
    //compute mode cpu -> use update MA57 linear solver's matrix
    
    if(nullptr == linSys_) {
      
      index_type itnz = 0;
      for(index_type i=0; i<M_condensed_->m(); ++i) {

        for(index_type p=M_condensed_->i_row()[i]; p<M_condensed_->i_row()[i+1]; ++p) {
          const index_type j = M_condensed_->j_col()[p];
          if(i<=j) {
            itnz++; 
          }
        }
      }
      linSys_ = determine_and_create_linsys(nx, nineq, itnz);
    }

    assert(linSys_);
    auto* linSys = dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
    auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sys_matrix());
    assert(Msys);
    assert(Msys->m() == M_condensed_->m());

    index_type itnz=0;
    for(index_type i=0; i<Msys->m(); ++i) {
      for(index_type p=M_condensed_->i_row()[i]; p<M_condensed_->i_row()[i+1]; ++p) {
        const index_type j = M_condensed_->j_col()[p];
        if(i<=j) {
          Msys->i_row()[itnz] = i;
          Msys->j_col()[itnz] = j;
          Msys->M()[itnz] = M_condensed_->M()[p];
          itnz++; 
        }
      }
    }
  }
  nlp_->runStats.kkt.tmUpdateLinsys.stop();
  
  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "KKT_SPARSE_Condensed linsys: Low-level linear system size %d nnz %d\n",
                      nx, 
                      M_condensed_->numberOfNonzeros());
  }

  //write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_>=0) {
    // TODO csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nx, 0, nineq);
  }
  return true; 
}

bool hiopKKTLinSysCondensedSparse::solve_compressed_direct(hiopVector& rx,
                                                           hiopVector& rd,
                                                           hiopVector& ryc,
                                                           hiopVector& ryd,
                                                           hiopVector& dx,
                                                           hiopVector& dd,
                                                           hiopVector& dyc,
                                                           hiopVector& dyd)
{
  assert(nlpSp_);
  assert(HessSp_);
  assert(Jac_dSp_);
  assert(0 == ryc.get_size() && "this KKT does not support equality constraints");

  size_type nx = rx.get_size();
  size_type nd = rd.get_size();
  size_type nyd = ryd.get_size();

  assert(rhs_);
  assert(rhs_->get_size() == nx);

   /* (H+Dx+Jd^T*(Dd+delta_wd*I)*Jd)dx = rx + Jd^T*Dd*ryd + Jd^T*rd
   * dd = Jd*dx - ryd
   * dyd = (Dd+delta_wd*I)*dd - rd = (Dd+delta_wd*I)*Jd*dx - (Dd+delta_wd*I)*ryd - rd
   */
  rhs_->copyFrom(rx);

  //working buffers in the size of nineq/nd using output as storage
  hiopVector& Dd_x_ryd = dyd;
  Dd_x_ryd.copyFrom(ryd);
  Dd_x_ryd.componentMult(*Hd_);

  hiopVector& DD_x_ryd_plus_rd = Dd_x_ryd;
  DD_x_ryd_plus_rd.axpy(1.0, rd);

  Jac_dSp_->transTimesVec(1.0, *rhs_, 1.0, DD_x_ryd_plus_rd);

  //
  // solve
  //
  bool linsol_ok = linSys_->solve(*rhs_);
  
  if(false==linsol_ok) {
    return false;
  }
  dx.copyFrom(*rhs_);

  dd.copyFrom(ryd);
  Jac_dSp_->timesVec(-1.0, dd, 1.0, dx);

  dyd.copyFrom(dd);
  dyd.componentMult(*Hd_);
  dyd.axpy(-1.0, rd);
  return true;
}

bool hiopKKTLinSysCondensedSparse::solveCompressed(hiopVector& rx,
                                                   hiopVector& rd,
                                                   hiopVector& ryc,
                                                   hiopVector& ryd,
                                                   hiopVector& dx,
                                                   hiopVector& dd,
                                                   hiopVector& dyc,
                                                   hiopVector& dyd)
{
  assert(nlpSp_);
  assert(HessSp_);
  assert(Jac_dSp_);
  assert(0 == ryc.get_size() && "this KKT does not support equality constraints");

  bool bret;

  nlp_->runStats.kkt.tmSolveInner.start();
  
  size_type nx = rx.get_size();
  size_type nd = rd.get_size();
  size_type nyd = ryd.get_size();

  // this is rhs used by the direct "condensed" solve
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
  }
  assert(rhs_->get_size() == nx);

  nlp_->log->write("RHS KKT_SPARSE_Condensed rx: ", rx,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed rd: ", rd,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryd:", ryd, hovIteration);

  bret = solve_compressed_direct(rx, rd, ryc, ryd, dx, dd, dyc, dyd);
  nlp_->runStats.kkt.tmSolveInner.stop();

  // Code for iterative refinement of the XDYcYd KKT system was removed since the parent
  // KKT class performs this now. Old code residing in this class can be found at
  // header file: https://github.com/LLNL/hiop/blob/fa61c1993128afd65a3cb21301c1f131922ceef8/src/Optimization/hiopKKTLinSysSparseCondensed.hpp#L209
  // implementation file: https://github.com/LLNL/hiop/blob/fa61c1993128afd65a3cb21301c1f131922ceef8/src/Optimization/hiopKKTLinSysSparseCondensed.cpp#L731

  
  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "(summary for linear solver from KKT_SPARSE_Condensed(direct))\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }

  
  nlp_->log->write("SOL KKT_SPARSE_Condensed dx: ", dx,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dd: ", dd,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyd:", dyd, hovMatrices);
  return bret;
}


hiopLinSolverSymSparse*
hiopKKTLinSysCondensedSparse::determine_and_create_linsys(size_type nx, size_type nineq, size_type nnz)
{   
  if(linSys_) {
    return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
  }
 
  int n = nx;
  auto linsolv = nlp_->options->GetString("linear_solver_sparse");
  if(nlp_->options->GetString("compute_mode") == "cpu") {

    //TODO:
    //add support for linear_solver == "cholmod"
    // maybe add pardiso as an option in the future
    //
    assert((linsolv=="ma57" || linsolv=="auto") && "Only MA57 or auto is supported on cpu.");
    
#ifdef HIOP_USE_COINHSL
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_Condensed linsys: alloc MA57 for matrix of size %d (0 cons)\n", n);
    linSys_ = new hiopLinSolverSymSparseMA57(n, nnz, nlp_);
#else
    assert(false && "HiOp was built without a sparse linear solver needed by the condensed KKT approach");
#endif // HIOP_USE_COINHSL
    
  } else {
    //
    // on device: compute_mode is hybrid, auto, or gpu
    //
    assert(nullptr==linSys_);

    assert((linsolv=="cusolver-chol" || linsolv=="auto") && "Only MA57 or auto is supported on cpu.");
    
#ifdef HIOP_USE_CUDA
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_Condensed linsys: alloc cuSOLVER-chol matrix size %d\n", n);
    assert(M_condensed_);
    linSys_ = new hiopLinSolverCholCuSparse(M_condensed_, nlp_);

#endif    
    
    //Return NULL (and assert) if a GPU sparse linear solver is not present
    assert(linSys_!=nullptr &&
           "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
           "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to 'cpu'");
  }
  
  assert(linSys_&& "KKT_SPARSE_XYcYd linsys: cannot instantiate backend linear solver");
  return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
}

} // end of namespace
