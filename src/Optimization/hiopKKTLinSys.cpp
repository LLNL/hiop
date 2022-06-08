// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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

#include "hiopKKTLinSys.hpp"
#include "hiopLinAlgFactory.hpp"
#include "hiop_blasdefs.hpp"

#include <cmath>

namespace hiop
{

hiopKKTLinSys::hiopKKTLinSys(hiopNlpFormulation* nlp)
  : nlp_(nlp),
    iter_(NULL),
    grad_f_(NULL),
    Jac_c_(NULL),
    Jac_d_(NULL),
    Hess_(NULL),
    perturb_calc_(NULL),
    safe_mode_(true),
    kkt_opr_(nullptr),
    prec_opr_(nullptr),
    ir_rhs_(nullptr),
    ir_x0_(nullptr),
    bicgIR_(nullptr),
    delta_wx_(nullptr),
    delta_wd_(nullptr),
    delta_cc_(nullptr),
    delta_cd_(nullptr)
    
{
  perf_report_ = "on"==hiop::tolower(nlp_->options->GetString("time_kkt"));
  mu_ = nlp_->options->GetNumeric("mu0");
  delta_wx_ = nlp_->alloc_primal_vec();
  delta_wd_ = nlp_->alloc_dual_ineq_vec();
  delta_cc_ = nlp_->alloc_dual_eq_vec();
  delta_cd_ = nlp_->alloc_dual_ineq_vec();
  delta_wx_->setToZero();
  delta_wd_->setToZero();
  delta_cc_->setToZero();
  delta_cd_->setToZero();
}

hiopKKTLinSys::~hiopKKTLinSys()
{
  delete kkt_opr_;
  delete prec_opr_;
  delete ir_rhs_;
  delete ir_x0_;
  delete bicgIR_;
  delete delta_wx_;
  delete delta_wd_;
  delete delta_cc_;
  delete delta_cd_;
}

//computes the solve error for the KKT Linear system; used only for correctness checking
double hiopKKTLinSys::errorKKT(const hiopResidual* resid, const hiopIterate* sol)
{
  nlp_->log->printf(hovLinAlgScalars, "KKT LinSys::errorKKT KKT_large residuals norm:\n");

  if(perturb_calc_) {
    perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);  
  } else {
    delta_wx_->setToZero();
    delta_wd_->setToZero();
    delta_cc_->setToZero();
    delta_cd_->setToZero();
  }

  double derr=1e20, aux;
  hiopVector *RX=resid->rx->new_copy();

  //RX = rx-H*dx-J'c*dyc-J'*dyd +dzl-dzu
  HessianTimesVec_noLogBarrierTerm(1.0, *RX, -1.0, *sol->x);
  RX->axzpy(-1., *delta_wx_, *sol->x);

  Jac_c_->transTimesVec(1.0, *RX, -1.0, *sol->yc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, *sol->yd);
  RX->axpy( 1.0, *sol->zl);
  RX->axpy(-1.0, *sol->zu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rx=%g\n", aux);


  //RD = rd - (-dyd - dvl + dvu + delta_wd_*dd)
  hiopVector* RD = resid->rd->new_copy();
  RD->axpy(+1., *sol->yd);
  RD->axpy(+1., *sol->vl);
  RD->axpy(-1., *sol->vu);
  RD->axzpy(-1., *delta_wd_, *sol->d);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rd=%g\n", aux);

  //RYC = ryc - Jc*dx + delta_cc_*dyc
  hiopVector* RYC=resid->ryc->new_copy();
  Jac_c_->timesVec(1.0, *RYC, -1.0, *sol->x);
  RYC->axzpy(1., *delta_cc_, *sol->yc);
  aux=RYC->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- ryc=%g\n", aux);
  delete RYC;

  //RYD=ryd - Jd*dx + dd + delta_cd_*dyd
  hiopVector* RYD=resid->ryd->new_copy();
  Jac_d_->timesVec(1.0, *RYD, -1.0, *sol->x);
  RYD->axpy(1.0, *sol->d);
  RYD->axzpy(1., *delta_cd_, *sol->yd);
  aux=RYD->infnorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- ryd=%g\n", aux);
  delete RYD;

  //RXL=rxl+x-sxl
  RX->copyFrom(*resid->rxl);
  RX->axpy( 1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxl);
  RX->selectPattern(nlp_->get_ixl());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rxl=%g\n", aux);
  //RXU=rxu-x-sxu
  RX->copyFrom(*resid->rxu);
  RX->axpy(-1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxu);
  RX->selectPattern(nlp_->get_ixu());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rxu=%g\n", aux);

  //RDL=rdl+d-sdl
  RD->copyFrom(*resid->rdl);
  RD->axpy( 1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdl);
  RD->selectPattern(nlp_->get_idl());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rdl=%g\n", aux);

  //RDU=rdu-d-sdu
  RD->copyFrom(*resid->rdu);
  RD->axpy(-1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdu);
  RD->selectPattern(nlp_->get_idu());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rdu=%g\n", aux);

  //complementarity residuals checks: rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszl);
  RX->axzpy(-1.0,*iter_->sxl,*sol->zl);
  RX->axzpy(-1.0,*iter_->zl, *sol->sxl);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rszl=%g\n", aux);
  //rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszu);
  RX->axzpy(-1.0,*iter_->sxu,*sol->zu);
  RX->axzpy(-1.0,*iter_->zu, *sol->sxu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rszu=%g\n", aux);
  delete RX; RX=NULL;

  //complementarity residuals checks: rsvl - Sdl dvl - Vl dsdl
  RD->copyFrom(*resid->rsvl);
  RD->axzpy(-1.0,*iter_->sdl,*sol->vl);
  RD->axzpy(-1.0,*iter_->vl, *sol->sdl);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rsvl=%g\n", aux);
  //complementarity residuals checks: rsvu - Sdu dvu - Vu dsdu
  RD->copyFrom(*resid->rsvu);
  RD->axzpy(-1.0,*iter_->sdu,*sol->vu);
  RD->axzpy(-1.0,*iter_->vu, *sol->sdu);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rsvu=%g\n", aux);
  delete RD;

  return derr;
}

bool hiopKKTLinSys::compute_directions_for_full_space(const hiopResidual* resid,
                                                      hiopIterate* dir)
{
  nlp_->runStats.kkt.tmSolveRhsManip.start();
  const hiopResidual &r=*resid;

  /***********************************************************************
   * compute the rest of the directions
   *
   */
  //dsxl = rxl + dx  and dzl= [Sxl]^{-1} ( - Zl*dsxl + rszl)
  if(nlp_->n_low_local()) {
    dir->sxl->copyFrom(*r.rxl);
    dir->sxl->axpy( 1.0,*dir->x);
    dir->sxl->selectPattern(nlp_->get_ixl());

    dir->zl->copyFrom(*r.rszl);
    dir->zl->axzpy(-1.0,*iter_->zl,*dir->sxl);
    dir->zl->componentDiv_w_selectPattern(*iter_->sxl, nlp_->get_ixl());
  } else {
    dir->sxl->setToZero();
    dir->zl->setToZero();
  }

  //dir->sxl->print();
  //dir->zl->print();
  //dsxu = rxu - dx and dzu = [Sxu]^{-1} ( - Zu*dsxu + rszu)
  if(nlp_->n_upp_local()) {
    dir->sxu->copyFrom(*r.rxu);
    dir->sxu->axpy(-1.0,*dir->x);
    dir->sxu->selectPattern(nlp_->get_ixu());

    dir->zu->copyFrom(*r.rszu);
    dir->zu->axzpy(-1.0,*iter_->zu,*dir->sxu);
    dir->zu->selectPattern(nlp_->get_ixu());
    dir->zu->componentDiv_w_selectPattern(*iter_->sxu, nlp_->get_ixu());
  } else {
    dir->sxu->setToZero();
    dir->zu->setToZero();
  }

  //dir->sxu->print();
  //dir->zu->print();
  //dsdl = rdl + dd and dvl = [Sdl]^{-1} ( - Vl*dsdl + rsvl)
  if(nlp_->m_ineq_low()) {
    dir->sdl->copyFrom(*r.rdl);
    dir->sdl->axpy( 1.0,*dir->d);
    dir->sdl->selectPattern(nlp_->get_idl());

    dir->vl->copyFrom(*r.rsvl);
    dir->vl->axzpy(-1.0,*iter_->vl,*dir->sdl);
    dir->vl->selectPattern(nlp_->get_idl());
    dir->vl->componentDiv_w_selectPattern(*iter_->sdl, nlp_->get_idl());
  } else {
    dir->sdl->setToZero();
    dir->vl->setToZero();
  }

  //dsdu = rdu - dd and dvu = [Sdu]^{-1} ( - Vu*dsdu + rsvu )
  if(nlp_->m_ineq_upp()>0) {
    dir->sdu->copyFrom(*r.rdu);
    dir->sdu->axpy(-1.0,*dir->d);
    dir->sdu->selectPattern(nlp_->get_idu());

    dir->vu->copyFrom(*r.rsvu);
    dir->vu->axzpy(-1.0,*iter_->vu,*dir->sdu);
    dir->vu->selectPattern(nlp_->get_idu());
    dir->vu->componentDiv_w_selectPattern(*iter_->sdu, nlp_->get_idu());
  } else {
    dir->sdu->setToZero();
    dir->vu->setToZero();
  }
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  assert(dir->sxl->matchesPattern(nlp_->get_ixl()));
  assert(dir->sxu->matchesPattern(nlp_->get_ixu()));
  assert(dir->sdl->matchesPattern(nlp_->get_idl()));
  assert(dir->sdu->matchesPattern(nlp_->get_idu()));
  assert(dir->zl->matchesPattern(nlp_->get_ixl()));
  assert(dir->zu->matchesPattern(nlp_->get_ixu()));
  assert(dir->vl->matchesPattern(nlp_->get_idl()));
  assert(dir->vu->matchesPattern(nlp_->get_idu()));

  //CHECK THE SOLUTION
  errorKKT(resid,dir);
  nlp_->runStats.kkt.tmResid.stop();
#endif

  return true;
}

int hiopKKTLinSysCurvCheck::factorizeWithCurvCheck()
{
  return linSys_->matrixChanged();
}

bool hiopKKTLinSysCurvCheck::factorize()
{
  assert(nlp_);

  // factorization + inertia correction if needed
  const size_t max_refactorization = 10;
  size_t num_refactorization = 0;
  int continue_re_fact;

  if(!perturb_calc_->compute_initial_deltas(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_)) {
    nlp_->log->printf(hovWarning, "linsys: Regularization perturbation on new linsys failed.\n");
    return false;
  }

  while(num_refactorization <= max_refactorization) {
#ifdef HIOP_DEEPCHECKS
    assert(delta_wx_->is_equal(*delta_wd_) && "something went wrong with IC");
    assert(delta_cc_->is_equal(*delta_cd_) && "something went wrong with IC");
#endif
      if(delta_wx_->get_size() == 1 && delta_cc_->get_size() == 1) {
        nlp_->log->printf(hovScalars, "linsys: delta_w=%12.5e delta_c=%12.5e (ic %d)\n",
                          delta_wx_->local_data_host()[0], delta_cc_->local_data_host()[0], num_refactorization);  
      }


    // the update of the linear system, including IC perturbations
    this->build_kkt_matrix(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);

    nlp_->runStats.kkt.tmUpdateInnerFact.start();

    // factorization
    int n_neg_eig = factorizeWithCurvCheck();

    nlp_->runStats.kkt.tmUpdateInnerFact.stop();

    continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, n_neg_eig, *delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);
    
    if(-1==continue_re_fact) {
      return false;
    } else if(0==continue_re_fact) {
      break;
    }

    // will do an inertia correction
    num_refactorization++;
    nlp_->runStats.kkt.nUpdateICCorr++;
  } // end of IC loop

  if(num_refactorization>max_refactorization) {
    nlp_->log->printf(hovError,
        "Reached max number (%d) of refactorization within an outer iteration.\n",
              max_refactorization);
    return false;
  }
  return true;
}

bool hiopKKTLinSysCurvCheck::factorize_inertia_free()
{
  assert(nlp_);

  int non_singular_mat = 1;
  int continue_re_fact;

  perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);

  continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, non_singular_mat, *delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_, true);

#ifdef HIOP_DEEPCHECKS
    assert(delta_wx_->is_equal(*delta_wd_) && "something went wrong with IC");
    assert(delta_cc_->is_equal(*delta_cd_) && "something went wrong with IC");
#endif
  if(delta_wx_->get_size() == 1 && delta_cc_->get_size() == 1) {
    nlp_->log->printf(hovScalars, "linsys: delta_w=%12.5e delta_c=%12.5e \n",
                      delta_wx_->local_data_host()[0], delta_cc_->local_data_host()[0]);  
  }
      
  // the update of the linear system, including IC perturbations
  this->build_kkt_matrix(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);

  nlp_->runStats.kkt.tmUpdateInnerFact.start();

  // factorization
  int solver_flag = factorizeWithCurvCheck();
  
  // if solver_flag<0, matrix becomes singular, or not pd (in condensed system) after adding regularization
  // this should not happen, but some linear solver may have numerical difficulty.
  // adding more regularization till it succeeds
  const size_t max_refactorization = 10;
  size_t num_refactorization = 0;

  while(num_refactorization<=max_refactorization && solver_flag < 0) {
    nlp_->log->printf(hovWarning, "linsys: matrix becomes singular after adding primal regularization!\n");

    continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, solver_flag, *delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);
    
    if(-1==continue_re_fact) {
      return false;
    } else {
      // this while loop is used to correct singularity
      assert(1==continue_re_fact);
    }
      
    
    if(delta_wx_->get_size() == 1 && delta_cc_->get_size() == 1) {
      nlp_->log->printf(hovScalars, "linsys: delta_w=%12.5e delta_c=%12.5e \n", delta_wx_->local_data_host()[0], delta_cc_->local_data_host()[0]);  
    }
  
    // the update of the linear system, including IC perturbations
    this->build_kkt_matrix(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);

    nlp_->runStats.kkt.tmUpdateInnerFact.start();

    // factorization
    solver_flag = factorizeWithCurvCheck();

    nlp_->runStats.kkt.tmUpdateInnerFact.stop();

    // will do an inertia correction
    num_refactorization++;
    nlp_->runStats.kkt.nUpdateICCorr++;
  } // end of IC loop

  nlp_->runStats.kkt.tmUpdateInnerFact.stop();

  return true;
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressed
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
bool hiopKKTLinSysCompressed::test_direction(const hiopIterate* dir, hiopMatrix* Hess)
{
  bool retval;
  nlp_->runStats.tmSolverInternal.start();

  if(!x_wrk_) {
    x_wrk_ = nlp_->alloc_primal_vec();
    x_wrk_->setToZero();
  }
  if(!d_wrk_) {
    d_wrk_ = nlp_->alloc_dual_ineq_vec();
    d_wrk_->setToZero();
  }

  hiopVector* sol_x = dir->get_x();
  hiopVector* sol_d = dir->get_d();
  double dWd = 0;
  double xs_nrmsq = 0.0;
  double dbl_wrk;
  perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);

  /* compute xWx = x(H+Dx_)x (for primal var [x,d] */
  Hess_->timesVec(0.0, *x_wrk_, 1.0, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);
  
  x_wrk_->copyFrom(*sol_x);
  x_wrk_->componentMult(*Dx_);
  x_wrk_->axzpy(1., *delta_wx_, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  d_wrk_->copyFrom(*sol_d);
  d_wrk_->componentMult(*Dd_);
  d_wrk_->axzpy(1., *delta_wd_, *sol_d);
  dWd += d_wrk_->dotProductWith(*sol_d);

  /* compute rhs for the dWd test */
  dbl_wrk = sol_x->twonorm();
  xs_nrmsq += dbl_wrk*dbl_wrk;
  dbl_wrk = sol_d->twonorm();
  xs_nrmsq += dbl_wrk*dbl_wrk;

  if(dWd < xs_nrmsq * nlp_->options->GetNumeric("neg_curv_test_fact")) {
    // have negative curvature. Add regularization and re-factorize the matrix
    retval = false;
  } else {
    // have positive curvature. Accept this factoraizaiton and direction.
    retval = true;
  }

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressedXYcYd
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/** 
 * Provides the functionality for reducing the KKT linear system to the
 * compressed linear below in dx, dd, dyc, and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions.
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * Relies on the pure virtual 'solveCompressed' to solve the compressed linear system
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 */
hiopKKTLinSysCompressedXYcYd::hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressed(nlp)
{
  Dd_inv_ = dynamic_cast<hiopVector*>(nlp_->alloc_dual_ineq_vec());
  assert(Dd_inv_ != NULL);

  ryd_tilde_ = Dd_inv_->alloc_clone();
}

hiopKKTLinSysCompressedXYcYd::~hiopKKTLinSysCompressedXYcYd()
{
  delete Dd_inv_;
  delete ryd_tilde_;
}

bool hiopKKTLinSysCompressedXYcYd::update(const hiopIterate* iter,
                                          const hiopVector* grad_f,
                                          const hiopMatrix* Jac_c,
                                          const hiopMatrix* Jac_d,
                                          hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.start_linsolve();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();

  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c; Jac_d_ = Jac_d;
  Hess_=Hess;

  int nx  = Hess_->m();
  assert(nx==Hess_->n()); assert(nx==Jac_c_->n()); assert(nx==Jac_d_->n());
  int neq = Jac_c_->m(), nineq = Jac_d_->m();

  //compute and put the barrier diagonals in
  //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
    assert(true==Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();

  //factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysCompressedXYcYd::computeDirections(const hiopResidual* resid,
                                                     hiopIterate* dir)
{ 
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual &r=*resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde  = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * ryd_tilde = ryd + [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}*
   *                     [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)]
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
   * Dd_inv = [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}
   * yd_tilde = ryd + Dd_inv*rd_tilde
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local()>0) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector&rl=*(dir->x);//temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    //rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern( 1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local()>0) {
    //ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector&ru=*(dir->x);//temporary working buffer
    ru.copyFrom(*r.rszu); ru.axzpy(-1.0,*iter_->zu, *r.rxu);
    //rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  //for ryd_tilde:
  ryd_tilde_->copyFrom(*r.ryd);
  // 1. the diag (Sdl^{-1}Vl+Sdu^{-1}Vu)^{-1} has already computed in Dd_inv in 'update'
  // 2. compute the left multiplicand in ryd2 (using buffer dir->sdl), that is
  //   ryd2 = [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)] (this is \tilde{r}_d in the notes)
  //    Inner ops are performed by accumulating in rd2  (buffer dir->sdu)
  hiopVector&ryd2=*dir->sdl;
  ryd2.copyFrom(*r.rd);

  if(nlp_->m_ineq_low()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    //ryd2 +=  Sdl^{-1}*(rsvl-Vl*rdl)
    ryd2.axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    //ryd2 += -Sdu^{-1}(rsvu-Vu*rdu)
    ryd2.axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }

  nlp_->log->write("Dinv (in computeDirections)", *Dd_inv_, hovMatrices);

  //now the final ryd_tilde += Dd^{-1}*ryd2
  ryd_tilde_->axzpy(1.0, ryd2, *Dd_inv_);
  
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  hiopVector* rx_tilde_save=rx_tilde_->new_copy();
  hiopVector* ryc_save=r.ryc->new_copy();
  hiopVector* ryd_tilde_save=ryd_tilde_->new_copy();
  nlp_->runStats.kkt.tmResid.stop();
#endif

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*rx_tilde_, *r.ryc, *ryd_tilde_, *dir->x, *dir->yc, *dir->yd);
  
  nlp_->runStats.kkt.tmSolveRhsManip.start();
  //recover dir->d = (D)^{-1}*(dir->yd + ryd2)
  dir->d->copyFrom(ryd2);
  dir->d->axpy(1.0,*dir->yd);
  dir->d->componentMult(*Dd_inv_);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  //dir->d->print();

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  errorCompressedLinsys(*rx_tilde_save,*ryc_save,*ryd_tilde_save, *dir->x, *dir->yc, *dir->yd);
  delete rx_tilde_save;
  delete ryc_save;
  delete ryd_tilde_save;
  nlp_->runStats.kkt.tmResid.stop();
#endif

  if(false==sol_ok) {
    return false;
  }

  bool bret = compute_directions_for_full_space(resid, dir);  

  nlp_->runStats.tmSolverInternal.stop();
  nlp_->runStats.linsolv.end_linsolve();
  return true;
}

#ifdef HIOP_DEEPCHECKS
//this method needs a bit of revisiting if becomes critical (mainly avoid dynamic allocations)
double hiopKKTLinSysCompressedXYcYd::
errorCompressedLinsys(const hiopVector& rx, const hiopVector& ryc, const hiopVector& ryd,
		      const hiopVector& dx, const hiopVector& dyc, const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXYcYd::errorCompressedLinsys residuals norm:\n");

  if(perturb_calc_) {
    perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);  
  } else {
    delta_wx_->setToZero();
    delta_wd_->setToZero();
    delta_cc_->setToZero();
    delta_cd_->setToZero();
  }

  double derr=1e20, aux;
  hiopVector *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess_->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx_, dx);
  RX->axzpy(-1., *delta_wx_, dx);

  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX; RX=NULL;

  hiopVector* RC=ryc.new_copy();
  Jac_c_->timesVec(1.0, *RC, -1.0, dx);
  RC->axzpy(1., *delta_cc_, dyc);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC; RC=NULL;

  hiopVector* RD=ryd.new_copy();
  Jac_d_->timesVec(1.0, *RD, -1.0, dx);

  RD->axzpy(1.0, *Dd_inv_, dyd);
  RD->axzpy(1., *delta_cd_, dyd);
  aux = RD->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RD;

  return derr;
}
#endif

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressedXDYcYd
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/* Provides the functionality for reducing the KKT linear system system to the
 * compressed linear below in dx, dd, dyc, and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions.
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
 * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
 * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]
 * and then to compute the rest of the search directions
 */
hiopKKTLinSysCompressedXDYcYd::hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressed(nlp)
{
//  Dd_ = dynamic_cast<hiopVector*>(nlp_->alloc_dual_ineq_vec());
//  assert(Dd_ != NULL);

  rd_tilde_ = Dd_->alloc_clone();
}

hiopKKTLinSysCompressedXDYcYd::~hiopKKTLinSysCompressedXDYcYd()
{
//  delete Dd_;
  delete rd_tilde_;
}

bool hiopKKTLinSysCompressedXDYcYd::update( const hiopIterate* iter,
                                            const hiopVector* grad_f,
                                            const hiopMatrix* Jac_c,
                                            const hiopMatrix* Jac_d,
                                            hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.start_linsolve();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();
  
  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c; Jac_d_ = Jac_d;

  Hess_=Hess;

  int nx  = Hess_->m(); assert(nx==Hess_->n()); assert(nx==Jac_c_->n()); assert(nx==Jac_d_->n());
  int neq = Jac_c_->m(), nineq = Jac_d_->m();

  // compute barrier diagonals (these change only between outer optimiz iterations)
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  assert(true==Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();
  
  //factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}


bool hiopKKTLinSysCompressedXDYcYd::computeDirections(const hiopResidual* resid, 
						      hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual &r=*resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector &rl=*(dir->x);//temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    //rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern( 1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local()) {
    //ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector &ru=*(dir->x);//temporary working buffer
    ru.copyFrom(*r.rszu); ru.axzpy(-1.0,*iter_->zu, *r.rxu);
    //rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  //for rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
  rd_tilde_->copyFrom(*r.rd);
  if(nlp_->m_ineq_low()) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    //rd_tilde +=  Sdl^{-1}*(rsvl-Vl*rdl)
    rd_tilde_->axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    //rd_tilde += -Sdu^{-1}(rsvu-Vu*rdu)
    rd_tilde_->axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }
  nlp_->log->write("Dd (in computeDirections)", *Dd_, hovMatrices);
  
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  hiopVector* rx_tilde_save = rx_tilde_->new_copy();
  hiopVector* rd_tilde_save = rd_tilde_->new_copy();
  hiopVector* ryc_save = r.ryc->new_copy();
  hiopVector* ryd_save = r.ryd->new_copy();
  nlp_->runStats.kkt.tmResid.stop();
#endif
  
  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*rx_tilde_, *rd_tilde_, *r.ryc, *r.ryd, *dir->x, *dir->d, *dir->yc, *dir->yd);
  
#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  double derr =
    errorCompressedLinsys(*rx_tilde_save, *rd_tilde_save, *ryc_save, *ryd_save,
			  *dir->x, *dir->d, *dir->yc, *dir->yd);
  if(derr>1e-8)
    nlp_->log->printf(hovWarning, "solve compressed high absolute resid norm (=%12.5e)\n", derr);
  delete rx_tilde_save;
  delete ryc_save;
  delete rd_tilde_save;
  delete ryd_save;
  nlp_->runStats.kkt.tmResid.stop();
#endif

  if(false == sol_ok) {
    nlp_->runStats.tmSolverInternal.stop();
    nlp_->runStats.linsolv.end_linsolve();
    return false;
  }
  
  bool bret = compute_directions_for_full_space(resid, dir);
  
  nlp_->runStats.tmSolverInternal.stop();
  nlp_->runStats.linsolv.end_linsolve();
  return true;
}


bool hiopKKTLinSys::compute_directions_w_IR(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  
  // skip IR if user set ir_outer_maxit to 0 or negative values
  if(0 >= nlp_->options->GetInteger("ir_outer_maxit")) {
    nlp_->runStats.tmSolverInternal.stop();
    return computeDirections(resid,dir);
  }
  const hiopResidual &r=*resid;

  // in the order of rx, rd, ryc, ryd, rxl, rxu, rdl, rdu, rszl, rszu, rsvl, rsvu
  const size_type nx = resid->rx->get_local_size();
  const size_type nd = resid->rd->get_local_size();
  const size_type nyc = resid->ryc->get_local_size();
  const size_type nyd = resid->ryd->get_local_size();
  size_type dim_rhs = 5*nx + 5*nd + nyc + nyd;
  
  /***********************************************************************
   * solve the compressed system as a preconditioner
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/

  if(nullptr == kkt_opr_) {
    kkt_opr_ = new hiopMatVecKKTFullOpr(this, iter_, resid, dir);
    prec_opr_ = new hiopPrecondKKTOpr(this, iter_, resid, dir);
    ir_rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), dim_rhs);
    ir_x0_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), dim_rhs);
    bicgIR_ = new hiopBiCGStabSolver(dim_rhs, nlp_->options->GetString("mem_space"), kkt_opr_, prec_opr_);
  }

  // need to reset the pointer to the current iter, since the outer loop keeps swtiching between curr_iter and trial_iter
  kkt_opr_->reset_curr_iter(iter_);

  // form the rhs for the sparse linSys  
  nlp_->runStats.kkt.tmSolveRhsManip.start();
  resid->rx->copyToStarting(*ir_rhs_,   0);
  resid->rd->copyToStarting(*ir_rhs_,   nx);
  resid->ryc->copyToStarting(*ir_rhs_,  nx+nd);
  resid->ryd->copyToStarting(*ir_rhs_,  nx+nd+nyc);
  resid->rxl->copyToStarting(*ir_rhs_,  nx+nd+nyc+nyd);
  resid->rxu->copyToStarting(*ir_rhs_,  nx+nd+nyc+nyd+nx);
  resid->rdl->copyToStarting(*ir_rhs_,  nx+nd+nyc+nyd+nx+nx);
  resid->rdu->copyToStarting(*ir_rhs_,  nx+nd+nyc+nyd+nx+nx+nd);
  resid->rszl->copyToStarting(*ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd);
  resid->rszu->copyToStarting(*ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx);
  resid->rsvl->copyToStarting(*ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx+nx);
  resid->rsvu->copyToStarting(*ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx+nx+nd);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
  double tol = std::min(mu_*nlp_->options->GetNumeric("ir_outer_tol_factor"), nlp_->options->GetNumeric("ir_outer_tol_min"));
  bicgIR_->set_max_num_iter(nlp_->options->GetInteger("ir_outer_maxit"));
  bicgIR_->set_tol(tol);
  bicgIR_->set_x0(0.0);

  bool bret = bicgIR_->solve(*ir_rhs_);

  // assemble dir from ir solution  
  dir->x->startingAtCopyFromStartingAt(0,   *ir_rhs_, 0);
  dir->d->startingAtCopyFromStartingAt(0,   *ir_rhs_, nx);
  dir->yc->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd);
  dir->yd->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd+nyc);
  dir->sxl->startingAtCopyFromStartingAt(0, *ir_rhs_, nx+nd+nyc+nyd);
  dir->sxu->startingAtCopyFromStartingAt(0, *ir_rhs_, nx+nd+nyc+nyd+nx);
  dir->sdl->startingAtCopyFromStartingAt(0, *ir_rhs_, nx+nd+nyc+nyd+nx+nx);
  dir->sdu->startingAtCopyFromStartingAt(0, *ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd);
  dir->zl->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd);
  dir->zu->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx);
  dir->vl->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx+nx);
  dir->vu->startingAtCopyFromStartingAt(0,  *ir_rhs_, nx+nd+nyc+nyd+nx+nx+nd+nd+nx+nx+nd);

  nlp_->runStats.kkt.nIterRefinInner += bicgIR_->get_sol_num_iter();
  nlp_->runStats.kkt.tmSolveInner.stop();
  if(!bret) {
    nlp_->log->printf(hovWarning, "%s", bicgIR_->get_convergence_info().c_str());

    // accept the stpe since this is IR
    bret = true;
  }

  nlp_->runStats.tmSolverInternal.stop();
  nlp_->runStats.linsolv.end_linsolve();
  
  return bret;
}



#ifdef HIOP_DEEPCHECKS
double hiopKKTLinSysCompressedXDYcYd::
errorCompressedLinsys(const hiopVector& rx, const hiopVector& rd,
		      const hiopVector& ryc, const hiopVector& ryd,
		      const hiopVector& dx, const hiopVector& dd,
		      const hiopVector& dyc, const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXDYcYd::errorCompressedLinsys residuals norm:\n");

  if(perturb_calc_) {
    perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);  
  } else {
    delta_wx_->setToZero();
    delta_wd_->setToZero();
    delta_cc_->setToZero();
    delta_cd_->setToZero();
  }

  double derr=-1., aux;
  hiopVector *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess_->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx_, dx);
  RX->axzpy(-1,*delta_wx_, dx);
  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX;

  //RD = rd + dyd - Dd*dd
  hiopVector* RD=rd.new_copy();
  RD->axpy( 1., dyd);
  RD->axzpy(-1., *Dd_, dd);
  RD->axzpy(-1.,*delta_wd_, dd);
  aux=RD->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rd=%g\n", aux);
  delete RD;

  hiopVector* RC=ryc.new_copy();
  Jac_c_->timesVec(1.0, *RC, -1.0, dx);
  RC->axzpy(1., *delta_cc_, dyc);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC;

  //RYD = ryd+dyd - Jd*dx
  hiopVector* RYD=ryd.new_copy();
  Jac_d_->timesVec(1.0, *RYD, -1.0, dx);
  RYD->axpy(1.0, dd);
  RYD->axzpy(1., *delta_cd_, dyd);
  aux = RYD->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RYD; RYD=NULL;

  return derr;
}
#endif


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysLowRank
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

hiopKKTLinSysLowRank::hiopKKTLinSysLowRank(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressedXYcYd(nlp)
{
  nlpD = dynamic_cast<hiopNlpDenseConstraints*>(nlp_);

  _kxn_mat = nlpD->alloc_multivector_primal(nlpD->m()); //!opt
  assert("DEFAULT" == toupper(nlpD->options->GetString("mem_space")));
  N = LinearAlgebraFactory::create_matrix_dense(nlpD->options->GetString("mem_space"),
                                                nlpD->m(),
                                                nlpD->m());
#ifdef HIOP_DEEPCHECKS
  Nmat=N->alloc_clone();
#endif
  _k_vec1 = dynamic_cast<hiopVector*>(nlpD->alloc_dual_vec());
}

hiopKKTLinSysLowRank::~hiopKKTLinSysLowRank()
{
  if(N)         delete N;
#ifdef HIOP_DEEPCHECKS
  if(Nmat)      delete Nmat;
#endif
  if(_kxn_mat)  delete _kxn_mat;
  if(_k_vec1)   delete _k_vec1;
}

bool hiopKKTLinSysLowRank::
update(const hiopIterate* iter,
       const hiopVector* grad_f,
       const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d,
       hiopHessianLowRank* Hess)
{
  nlp_->runStats.tmSolverInternal.start();

  iter_=iter;
  grad_f_ = dynamic_cast<const hiopVector*>(grad_f);
  Jac_c_ = Jac_c; Jac_d_ = Jac_d;
  //Hess = dynamic_cast<hiopHessianInvLowRank*>(Hess_);
  Hess_=HessLowRank=Hess;

  //compute the diagonals
  //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  HessLowRank->updateLogBarrierDiagonal(*Dx_);

  //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_inv_->setToZero();
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
  assert(true==Dd_inv_->allPositive());
#endif
  Dd_->copyFrom(*Dd_inv_);
  Dd_inv_->invert();

  nlp_->runStats.tmSolverInternal.stop();

  nlp_->log->write("Dd_inv in KKT", *Dd_inv_, hovMatrices);
  return true;
}


/* Solves the system corresponding to directions for x, yc, and yd, namely
 * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx  ]
 * [    Jc          0     0     ] [dyc] = [ ryc ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd ]
 *
 * This is done by forming and solving
 * [ Jc*(H+Dx)^{-1}*Jc^T   Jc*(H+Dx)^{-1}*Jd^T          ] [dyc] = [ Jc(H+Dx)^{-1} rx - ryc ]
 * [ Jd*(H+Dx)^{-1}*Jc^T   Jd*(H+Dx)^{-1}*Jd^T + Dd^{-1}] [dyd]   [ Jd(H+dx)^{-1} rx - ryd ]
 * and then solving for dx from
 *  dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
 *
 * Note that ops H+Dx are provided by hiopHessianLowRank
 */
bool hiopKKTLinSysLowRank::
solveCompressed(hiopVector& rx, hiopVector& ryc, hiopVector& ryd,
		hiopVector& dx, hiopVector& dyc, hiopVector& dyd)
{
#ifdef HIOP_DEEPCHECKS
  //some outputing
  nlp_->log->write("KKT Low rank: solve compressed RHS", hovIteration);
  nlp_->log->write("  rx: ",  rx, hovIteration); nlp_->log->write(" ryc: ", ryc, hovIteration); nlp_->log->write(" ryd: ", ryd, hovIteration);
  nlp_->log->write("  Jc: ", *Jac_c_, hovMatrices);
  nlp_->log->write("  Jd: ", *Jac_d_, hovMatrices);
  nlp_->log->write("  Dd_inv: ", *Dd_inv_, hovMatrices);
  assert(Dd_inv_->isfinite_local() && "Something bad happened: nan or inf value");
#endif

  hiopMatrixDense& J = *_kxn_mat;
  const hiopMatrixDense* Jac_c_de = dynamic_cast<const hiopMatrixDense*>(Jac_c_); assert(Jac_c_de);
  const hiopMatrixDense* Jac_d_de = dynamic_cast<const hiopMatrixDense*>(Jac_d_); assert(Jac_d_de);
  J.copyRowsFrom(*Jac_c_de, nlp_->m_eq(), 0); //!opt
  J.copyRowsFrom(*Jac_d_de, nlp_->m_ineq(), nlp_->m_eq());//!opt

  //N =  J*(Hess\J')
  //Hess->symmetricTimesMat(0.0, *N, 1.0, J);
  HessLowRank->symMatTimesInverseTimesMatTrans(0.0, *N, 1.0, J);

  //subdiag of N += 1., Dd_inv
  N->addSubDiagonal(1., nlp_->m_eq(), *Dd_inv_);
#ifdef HIOP_DEEPCHECKS
  assert(J.isfinite());
  nlp_->log->write("solveCompressed: N is", *N, hovMatrices);
  nlp_->log->write("solveCompressed: rx is", rx, hovMatrices);
  nlp_->log->printf(hovLinAlgScalars, "inf norm of Dd_inv is %g\n", Dd_inv_->infnorm());
  N->assertSymmetry(1e-10);
#endif

  //compute the rhs of the lin sys involving N
  //  1. first compute (H+Dx)^{-1} rx_tilde and store it temporarily in dx
  HessLowRank->solve(rx, dx);
#ifdef HIOP_DEEPCHECKS
  assert(rx.isfinite_local() && "Something bad happened: nan or inf value");
  assert(dx.isfinite_local() && "Something bad happened: nan or inf value");
#endif

  // 2 . then rhs =   [ Jc(H+Dx)^{-1}*rx - ryc ]
  //                  [ Jd(H+dx)^{-1}*rx - ryd ]
  hiopVector& rhs=*_k_vec1;
  rhs.copyFromStarting(0, ryc);
  rhs.copyFromStarting(nlp_->m_eq(), ryd);
  J.timesVec(-1.0, rhs, 1.0, dx);

#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("solveCompressed: dx sol is", dx, hovMatrices);
  nlp_->log->write("solveCompressed: rhs for N is", rhs, hovMatrices);
  Nmat->copyFrom(*N);
  hiopVector* r=rhs.new_copy(); //save the rhs to check the norm of the residual
#endif

  //
  //solve N * dyc_dyd = rhs
  //
  int ierr = solveWithRefin(*N,rhs);
  //int ierr = solve(*N,rhs);

  hiopVector& dyc_dyd= rhs;
  dyc_dyd.copyToStarting(0,           dyc);
  dyc_dyd.copyToStarting(nlp_->m_eq(), dyd);

  //now solve for dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
  //first rx = -(Jc^T*dyc+Jd^T*dyd - rx)
  J.transTimesVec(1.0, rx, -1.0, dyc_dyd);
  //then dx = (H+Dx)^{-1} rx
  HessLowRank->solve(rx, dx);

#ifdef HIOP_DEEPCHECKS
  //some outputing
  nlp_->log->write("KKT Low rank: solve compressed SOL", hovIteration);
  nlp_->log->write("  dx: ",  dx, hovIteration); nlp_->log->write(" dyc: ", dyc, hovIteration); nlp_->log->write(" dyd: ", dyd, hovIteration);
  delete r;
#endif

  return ierr==0;
}

int hiopKKTLinSysLowRank::solveWithRefin(hiopMatrixDense& M, hiopVector& rhs)
{
  // 1. Solve dposvx (solve + equilibrating + iterative refinement + forward and backward error estimates)
  // 2. Check the residual norm
  // 3. If residual norm is not small enough, then perform iterative refinement. This is because dposvx
  // does not always provide a small enough residual since it stops (possibly without refinement) based on
  // the forward and backward estimates

  int N=M.n();
  if(N<=0) return 0;

  hiopMatrixDense* Aref = M.new_copy();
  hiopVector* rhsref = rhs.new_copy();

  char FACT='E';
  char UPLO='L';

  int NRHS=1;
  double* A=M.local_data();
  int LDA=N;
  double* AF=new double[N*N];
  int LDAF=N;
  char EQUED='N'; //it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB=N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, FERR, BERR;
  double* WORK = new double[3*N];
  int* IWORK = new int[N];
  int INFO;

  //
  // 1. solve
  //
  DPOSVX(&FACT, &UPLO, &N, &NRHS,
	 A, &LDA,
	 AF, &LDAF,
	 &EQUED,
	 S,
	 B, &LDB,
	 X, &LDX,
	 &RCOND, &FERR, &BERR,
	 WORK, IWORK,
	 &INFO);
  //printf("INFO ===== %d  RCOND=%g  FERR=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, FERR, BERR, EQUED);
  //
  // 2. check residual
  //
  hiopVector* x = rhs.alloc_clone();
  hiopVector* dx    = rhs.alloc_clone();
  hiopVector* resid = rhs.alloc_clone();
  int nIterRefin=0;double nrmResid;
  int info;
  const int MAX_ITER_REFIN=3;
  while(true) {
    x->copyFrom(X);
    resid->copyFrom(*rhsref);
    Aref->timesVec(1.0, *resid, -1.0, *x);

    nlp_->log->write("resid", *resid, hovLinAlgScalars);

    nrmResid= resid->infnorm();
    nlp_->log->printf(hovScalars, "hiopKKTLinSysLowRank::solveWithRefin iterrefin=%d  residual norm=%g\n", nIterRefin, nrmResid);

    if(nrmResid<1e-8) break;

    if(nIterRefin>=MAX_ITER_REFIN) {
      nlp_->log->write("N", *Aref, hovMatrices);
      nlp_->log->write("sol", *x, hovMatrices);
      nlp_->log->write("rhs", *rhsref, hovMatrices);

      nlp_->log->printf(hovWarning, "hiopKKTLinSysLowRank::solveWithRefin reduced residual to ONLY (inf-norm) %g after %d iterative refinements\n", nrmResid, nIterRefin);
      break;
      //assert(false && "too many refinements");
    }
    if(0) { //iter refin based on symmetric indefinite factorization+solve


      int _V_ipiv_vec[1000]; double _V_work_vec[1000]; int lwork=1000;
      M.copyFrom(*Aref);
      DSYTRF(&UPLO, &N, M.local_data(), &LDA, _V_ipiv_vec, _V_work_vec, &lwork, &info);
      assert(info==0);
      DSYTRS(&UPLO, &N, &NRHS, M.local_data(), &LDA, _V_ipiv_vec, resid->local_data(), &LDB, &info);
      assert(info==0);
    } else { //iter refin based on symmetric positive definite factorization+solve
      M.copyFrom(*Aref);
      //for(int i=0; i<4; i++) M.local_data()[i][i] +=1e-8;
      DPOTRF(&UPLO, &N, M.local_data(), &LDA, &info);
      if(info>0)
	nlp_->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf (Chol fact) "
			  "detected %d minor being indefinite.\n", info);
      else
	if(info<0)
	  nlp_->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf returned "
			    "error %d\n", info);

      DPOTRS(&UPLO,&N, &NRHS, M.local_data(), &LDA, resid->local_data(), &LDA, &info);
      if(info<0)
	nlp_->log->printf(hovError, "hiopKKTLinSysLowRank::solveWithFactors: dpotrs returned "
			  "error %d\n", info);
    }

   // //FACT='F'; EQUED='Y'; //reuse the factorization and the equilibration
   //  M.copyFrom(*Aref);
   //  A = M.local_buffer();
   //  dposvx_(&FACT, &UPLO, &N, &NRHS,
   // 	    A, &LDA,
   // 	    AF, &LDAF,
   // 	    &EQUED,
   // 	    S,
   // 	    resid.local_data(), &LDB,
   // 	    X, &LDX,
   // 	    &RCOND, &FERR, &BERR,
   // 	    WORK, IWORK,
   // 	    &INFO);
   //  printf("INFO ===== %d  RCOND=%g  FERR=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, FERR, BERR, EQUED);

    dx->copyFrom(*resid);
    x->axpy(1., *dx);

    nIterRefin++;
  }

  rhs.copyFrom(*x);
  delete[] AF;
  delete[] S;
  delete[] X;
  delete[] WORK;
  delete[] IWORK;
  delete Aref;
  delete rhsref;
  delete x;
  delete dx;
  delete resid;

// #ifdef HIOP_DEEPCHECKS
//   hiopVectorPar sol(rhs.get_size());
//   hiopVectorPar rhss(rhs.get_size());
//   sol.copyFrom(rhs); rhss.copyFrom(*r);
//   double relErr=solveError(*Nmat, rhs, *r);
//   if(relErr>1e-5)  {
//     nlp_->log->printf(hovWarning, "large rel. error (%g) in linear solver occured the Cholesky solve (hiopKKTLinSys)\n", relErr);

//     nlp_->log->write("matrix N=", *Nmat, hovError);
//     nlp_->log->write("rhs", rhss, hovError);
//     nlp_->log->write("sol", sol, hovError);

//     assert(false && "large error (%g) in linear solve (hiopKKTLinSys), equilibrating the matrix and/or iterative refinement are needed (see dposvx/x)");
//   } else
//     if(relErr>1e-16)
//       nlp_->log->printf(hovWarning, "considerable rel. error (%g) in linear solver occured the Cholesky solve (hiopKKTLinSys)\n", relErr);

//   nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysLowRank::solveCompressed: Cholesky solve: relative error %g\n", relErr);
//   delete r;
// #endif
  return 0;
}

int hiopKKTLinSysLowRank::solve(hiopMatrixDense& M, hiopVector& rhs)
{
  char FACT='E';
  char UPLO='L';
  int N=M.n();
  int NRHS=1;
  double* A=M.local_data();
  int LDA=N;
  double* AF=new double[N*N];
  int LDAF=N;
  char EQUED='N'; //it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB=N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, FERR, BERR;
  double* WORK = new double[3*N];
  int* IWORK = new int[N];
  int INFO;

  DPOSVX(&FACT, &UPLO, &N, &NRHS,
	 A, &LDA,
	 AF, &LDAF,
	 &EQUED,
	 S,
	 B, &LDB,
	 X, &LDX,
	 &RCOND, &FERR, &BERR,
	 WORK, IWORK,
	 &INFO);

  rhs.copyFrom(S);
  nlp_->log->write("Scaling S", rhs, hovSummary);

  //printf("INFO ===== %d  RCOND=%g  FERR=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, FERR, BERR, EQUED);

  rhs.copyFrom(X);
  delete [] AF;
  delete [] S;
  delete [] X;
  delete [] WORK;
  delete [] IWORK;
  return 0;
}

/* this code works fine but requires xblas
int hiopKKTLinSysLowRank::solveWithRefin(hiopMatrixDense& M, hiopVectorPar& rhs)
{
  char FACT='E';
  char UPLO='L';
  int N=M.n();
  int NRHS=1;
  double* A=M.local_buffer();
  int LDA=N;
  double* AF=new double[N*N];
  int LDAF=N;
  char EQUED='N'; //it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB=N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, BERR;
  double RPVGRW; //Reciprocal pivot growth
  int N_ERR_BNDS=3;
  double* ERR_BNDS_NORM = new double[NRHS*N_ERR_BNDS];
  double* ERR_BNDS_COMP = new double[NRHS*N_ERR_BNDS];
  int NPARAMS=3;
  double PARAMS[NPARAMS];
  PARAMS[0]=1.0;  //Use the extra-precise refinement algorithm
  PARAMS[1]=3.0; //Maximum number of residual computations allowed for refinement
  PARAMS[2]=1.0; //attempt to find a solution with small componentwise
  double* WORK = new double[4*N];
  int* IWORK = new int[N];
  int INFO;

  dposvxx_(&FACT, &UPLO, &N, &NRHS,
	   A, &LDA,
	   AF, &LDAF,
	   &EQUED,
	   S,
	   B, &LDB,
	   X, &LDX,
	   &RCOND, &RPVGRW, &BERR,
	   &N_ERR_BNDS, ERR_BNDS_NORM, ERR_BNDS_COMP,
	   &NPARAMS, PARAMS,
	   WORK, IWORK,
	   &INFO);

  //rhs.copyFrom(S);
  //nlp_->log->write("Scaling S", rhs, hovSummary);

  //M.copyFrom(AF);
  //nlp_->log->write("Factoriz ", M, hovSummary);

  printf("INFO ===== %d  RCOND=%g  RPVGRW=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, RPVGRW, BERR, EQUED);
  printf("               ERR_BNDS_NORM=%g %g %g    ERR_BNDS_COMP=%g %g %g \n", ERR_BNDS_NORM[0], ERR_BNDS_NORM[1], ERR_BNDS_NORM[2], ERR_BNDS_COMP[0], ERR_BNDS_COMP[1], ERR_BNDS_COMP[2]);
  printf("               PARAMS=%g %g %g \n", PARAMS[0], PARAMS[1], PARAMS[2]);


  rhs.copyFrom(X);
  delete [] AF;
  delete [] S;
  delete [] X;
  delete [] ERR_BNDS_NORM;
  delete [] ERR_BNDS_COMP;
  delete [] WORK;
  delete [] IWORK;
  return 0;
}
*/

#ifdef HIOP_DEEPCHECKS

double hiopKKTLinSysLowRank::
errorCompressedLinsys(const hiopVector& rx, const hiopVector& ryc, const hiopVector& ryd,
		      const hiopVector& dx, const hiopVector& dyc, const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysLowRank::errorCompressedLinsys residuals norm:\n");

  double derr=-1., aux;
  hiopVector *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  HessLowRank->timesVec(1.0, *RX, -1.0, dx);
  //RX->axzpy(-1.0,*Dx,dx);
  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>>  rx=%g\n", aux);
  // if(aux>1e-8) {
  // nlp_->log->write("Low rank Hessian is:", *Hess, hovLinAlgScalars);
  // }
  delete RX; RX=NULL;

  hiopVector* RC=ryc.new_copy();
  Jac_c_->timesVec(1.0,*RC, -1.0,dx);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>> ryc=%g\n", aux);
  delete RC; RC=NULL;

  hiopVector* RD=ryd.new_copy();
  Jac_d_->timesVec(1.0,*RD, -1.0, dx);
  RD->axzpy(1.0, *Dd_inv_, dyd);
  aux = RD->twonorm();
  derr=fmax(derr,aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>> ryd=%g\n", aux);
  delete RD; RD=NULL;

  return derr;
}

double hiopKKTLinSysLowRank::solveError(const hiopMatrixDense& M,  const hiopVector& x, hiopVector& rhs)
{
  double relError;
  double rhsnorm=rhs.infnorm();
  M.timesVec(1.0,rhs,-1.0,x);
  double resnorm=rhs.infnorm();

  relError=resnorm;// / (1+rhsnorm);
  return relError;
}
#endif


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysFull
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
bool hiopKKTLinSysFull::update(const hiopIterate* iter,
                               const hiopVector* grad_f,
                               const hiopMatrix* Jac_c,
                               const hiopMatrix* Jac_d,
                               hiopMatrix* Hess)
{
  
  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c; 
  Jac_d_ = Jac_d;
  Hess_=Hess;
  nlp_->runStats.linsolv.start_linsolve();
  nlp_->runStats.tmSolverInternal.start();

  // factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}



bool hiopKKTLinSysFull::computeDirections(const hiopResidual* resid,
						      hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();

  const hiopResidual &r=*resid;

  /***********************************************************************
   * solve the full system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solve(*r.rx, *r.ryc, *r.ryd, *r.rd,
                      *r.rdl, *r.rdu, *r.rxl, *r.rxu,
                      *r.rsvl, *r.rsvu, *r.rszl, *r.rszu,
                      *dir->x, *dir->yc, *dir->yd, *dir->d,
                      *dir->vl, *dir->vu, *dir->zl, *dir->zu,
                      *dir->sdl, *dir->sdu, *dir->sxl, *dir->sxu);

  nlp_->runStats.linsolv.end_linsolve();
  nlp_->runStats.tmSolverInternal.stop();
  return sol_ok;
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopMatVecKKTFullOpr
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
hiopMatVecKKTFullOpr::hiopMatVecKKTFullOpr(hiopKKTLinSys* kkt, 
                               const hiopIterate* iter,
                               const hiopResidual* resid,
                               const hiopIterate* dir)
    : kkt_(kkt),
      iter_(iter),
      resid_(nullptr),
      dir_(nullptr)
{
  resid_ = new hiopResidual(kkt_->nlp_);
  dir_ = new hiopIterate(kkt_->nlp_);
  dx_ = dir_->get_x(); 
  dx_->setToZero();
  dd_ = dir_->get_d();
  dd_->setToZero();
  dyc_ = dir_->get_yc();
  dyc_->setToZero();
  dyd_ = dir_->get_yd();
  dyd_->setToZero();
  dsxl_ = dir_->get_sxl();
  dsxl_->setToZero();
  dsxu_ = dir_->get_sxu();
  dsxu_->setToZero();
  dsdl_ = dir_->get_sdl();
  dsdl_->setToZero();
  dsdu_ = dir_->get_sdu();
  dsdu_->setToZero();
  dzl_ = dir_->get_zl();
  dzl_->setToZero();
  dzu_ = dir_->get_zu();
  dzu_->setToZero();
  dvl_ = dir_->get_vl();
  dvl_->setToZero();
  dvu_ = dir_->get_vu();
  dvu_->setToZero();

  yrx_ = resid_->get_rx();
  yrx_->setToZero();
  yrd_ = resid_->get_rd();
  yrd_->setToZero();
  yryc_ = resid_->get_ryc();
  yryc_->setToZero();
  yryd_ = resid_->get_ryd();
  yryd_->setToZero();
  yrsxl_ = resid_->get_rxl();
  yrsxl_->setToZero();
  yrsxu_ = resid_->get_rxu();
  yrsxu_->setToZero();
  yrsdl_ = resid_->get_rdl();
  yrsdl_->setToZero();
  yrsdu_ = resid_->get_rdu();
  yrsdu_->setToZero();
  yrzl_ = resid_->get_rszl();
  yrzl_->setToZero();
  yrzu_ = resid_->get_rszu();
  yrzu_->setToZero();
  yrvl_ = resid_->get_rsvl();
  yrvl_->setToZero();
  yrvu_ = resid_->get_rsvu();
  yrvu_->setToZero();
}

bool hiopMatVecKKTFullOpr::split_vec_to_build_it(const hiopVector& x)
{
  size_type nx = dx_->get_size();
  size_type neq = dyc_->get_size();
  size_type nineq = dyd_->get_size();

  dx_->startingAtCopyFromStartingAt(0,   x, 0);
  dd_->startingAtCopyFromStartingAt(0,   x, nx);
  dyc_->startingAtCopyFromStartingAt(0,  x, nx+nineq);
  dyd_->startingAtCopyFromStartingAt(0,  x, nx+nineq+neq);
  dsxl_->startingAtCopyFromStartingAt(0, x, nx+nineq+neq+nineq);
  dsxu_->startingAtCopyFromStartingAt(0, x, nx+nineq+neq+nineq+nx);
  dsdl_->startingAtCopyFromStartingAt(0, x, nx+nineq+neq+nineq+nx+nx);
  dsdu_->startingAtCopyFromStartingAt(0, x, nx+nineq+neq+nineq+nx+nx+nineq);
  dzl_->startingAtCopyFromStartingAt(0,  x, nx+nineq+neq+nineq+nx+nx+nineq+nineq);
  dzu_->startingAtCopyFromStartingAt(0,  x, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx);
  dvl_->startingAtCopyFromStartingAt(0,  x, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx);
  dvu_->startingAtCopyFromStartingAt(0,  x, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq);
  return true;
}

bool hiopMatVecKKTFullOpr::combine_res_to_build_vec(hiopVector& y)
{
  size_type nx = dx_->get_size();
  size_type neq = dyc_->get_size();
  size_type nineq = dyd_->get_size();

  yrx_->copyToStarting(   y, 0);
  yrd_->copyToStarting(   y, nx);
  yryc_->copyToStarting(  y, nx+nineq);
  yryd_->copyToStarting(  y, nx+nineq+neq);
  yrsxl_->copyToStarting( y, nx+nineq+neq+nineq);
  yrsxu_->copyToStarting( y, nx+nineq+neq+nineq+nx);
  yrsdl_->copyToStarting( y, nx+nineq+neq+nineq+nx+nx);
  yrsdu_->copyToStarting( y, nx+nineq+neq+nineq+nx+nx+nineq);
  yrzl_->copyToStarting(  y, nx+nineq+neq+nineq+nx+nx+nineq+nineq);
  yrzu_->copyToStarting(  y, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx);
  yrvl_->copyToStarting(  y, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx);
  yrvu_->copyToStarting(  y, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq);
  return true;
}

/**
 * Full KKT matrix is
 * [   H    0   Jc^T  Jd^T |  -I  I   0   0   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  0     0     0    -I  |  0   0  -I   I   |  0   0   0   0  ] [  dd]   [    rd    ]
 * [  Jc    0     0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    -I    0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [ -I     0     0     0  |  0   0   0   0   |  I   0   0   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0     0  |  0   0   0   0   |  0   I   0   0  ] [ dzu]   [   rxu    ]
 * [  0     -I    0     0  |  0   0   0   0   |  0   0   I   0  ] [ dvl]   [   rdl    ]
 * [  0     I     0     0  |  0   0   0   0   |  0   0   0   I  ] [ dvu]   [   rdu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0     0  | Sl^x 0   0   0   | Zl   0   0   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0     0  |  0  Su^x 0   0   |  0  Zu   0   0  ] [dsxu]   [  rszu    ]
 * [  0     0     0     0  |  0   0  Sl^d 0   |  0   0  Vl   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0     0  |  0   0   0  Su^d |  0   0   0  Vu  ] [dsdu]   [  rsvu    ]
 *
 * this method computes y = KKT * x
 */
bool hiopMatVecKKTFullOpr::times_vec(hiopVector& y, const hiopVector& x)
{
  const hiopMatrix* Jac_c = kkt_->Jac_c_;
  const hiopMatrix* Jac_d = kkt_->Jac_d_;
  const hiopMatrix* Hess = kkt_->Hess_;

  assert(kkt_->get_perturb_calc());
  hiopVector* delta_wx = kkt_->delta_wx_;
  hiopVector* delta_wd = kkt_->delta_wd_;
  hiopVector* delta_cc = kkt_->delta_cc_;
  hiopVector* delta_cd = kkt_->delta_cd_;
  if(kkt_->get_perturb_calc()) {
    kkt_->get_perturb_calc()->get_curr_perturbations(*delta_wx, *delta_wd, *delta_cc, *delta_cd);
  } else {
    delta_wx->setToZero();
    delta_wd->setToZero();
    delta_cc->setToZero();
    delta_cd->setToZero();
  }

  bool bret = split_vec_to_build_it(x);

  //rx = H*dx + delta_wx*I*dx + Jc'*dyc + Jd'*dyd - dzl + dzu
  Hess->timesVec(0.0, *yrx_, +1.0, *dx_);
  yrx_->axzpy(1., *delta_wx, *dx_);
  Jac_c->transTimesVec(1.0, *yrx_, 1.0, *dyc_);
  Jac_d->transTimesVec(1.0, *yrx_, 1.0, *dyd_);
  yrx_->axpy(-1.0, *dzl_);
  yrx_->axpy( 1.0, *dzu_);

  //RD = delta_wd_*dd - dyd - dvl + dvu
  yrd_->setToZero();
  yrd_->axpy(-1., *dyd_);
  yrd_->axpy(-1., *dvl_);
  yrd_->axpy(+1., *dvu_);
  yrd_->axzpy(1., *delta_wd, *dd_);

  //RYC = Jc*dx - delta_cc_*dyc
  Jac_c->timesVec(0.0, *yryc_, 1.0, *dx_);
  yryc_->axzpy(-1., *delta_cc, *dyc_);

  //RYD = Jd*dx - dd - delta_cd_*dyd
  Jac_d->timesVec(0.0, *yryd_, 1.0, *dx_);
  yryd_->axpy(-1.0, *dd_);
  yryd_->axzpy(-1., *delta_cd, *dyd_);

  //RXL = -dx + dsxl
  yrsxl_->copyFrom(*dsxl_);
  yrsxl_->axpy(-1.0, *dx_);
  yrsxl_->selectPattern(kkt_->nlp_->get_ixl());

  //RXU = dx + dsxu
  yrsxu_->copyFrom(*dsxu_);
  yrsxu_->axpy( 1.0, *dx_);
  yrsxu_->selectPattern(kkt_->nlp_->get_ixu());

  //RDL = -dd + dsdl
  yrsdl_->copyFrom(*dsdl_);
  yrsdl_->axpy(-1.0, *dd_);
  yrsdl_->selectPattern(kkt_->nlp_->get_idl());

  //RDU = dd + dsdu
  yrsdu_->copyFrom(*dsdu_);
  yrsdu_->axpy( 1.0, *dd_);
  yrsdu_->selectPattern(kkt_->nlp_->get_idu());

  // Sxl dzxl + Zxl dsxl
  yrzl_->setToZero();
  yrzl_->axzpy(1.0,*iter_->get_sxl(), *dzl_);
  yrzl_->axzpy(1.0,*iter_->get_zl(), *dsxl_);

  // Sxu dzxu + Zxu dsxu
  yrzu_->setToZero();
  yrzu_->axzpy(1.0,*iter_->get_sxu(),*dzu_);
  yrzu_->axzpy(1.0,*iter_->get_zu(), *dsxu_);

  // Sdl dzdl + Zdl dsdl
  yrvl_->setToZero();
  yrvl_->axzpy(1.0,*iter_->get_sdl(), *dvl_);
  yrvl_->axzpy(1.0,*iter_->get_vl(), *dsdl_);

  // Sdu dzdu + Zdu dsdu
  yrvu_->setToZero();
  yrvu_->axzpy(1.0,*iter_->get_sdu(),*dvu_);
  yrvu_->axzpy(1.0,*iter_->get_vu(), *dsdu_);

  bret = combine_res_to_build_vec(y);
  return true;
}

/**
 * Full KKT matrix is
 * [   H    0   Jc^T  Jd^T |  -I  I   0   0   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  0     0     0    -I  |  0   0  -I   I   |  0   0   0   0  ] [  dd]   [    rd    ]
 * [  Jc    0     0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    -I    0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [ -I     0     0     0  |  0   0   0   0   |  I   0   0   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0     0  |  0   0   0   0   |  0   I   0   0  ] [ dzu]   [   rxu    ]
 * [  0     -I    0     0  |  0   0   0   0   |  0   0   I   0  ] [ dvl]   [   rdl    ]
 * [  0     I     0     0  |  0   0   0   0   |  0   0   0   I  ] [ dvu]   [   rdu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0     0  | Sl^x 0   0   0   | Zl   0   0   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0     0  |  0  Su^x 0   0   |  0  Zu   0   0  ] [dsxu]   [  rszu    ]
 * [  0     0     0     0  |  0   0  Sl^d 0   |  0   0  Vl   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0     0  |  0   0   0  Su^d |  0   0   0  Vu  ] [dsdu]   [  rsvu    ]
 *
 * this method computes y = KKT' * x
 */
bool hiopMatVecKKTFullOpr::trans_times_vec(hiopVector& y, const hiopVector& x)
{
  // full KKT is not symmetric!
  const hiopMatrix* Jac_c = kkt_->Jac_c_;
  const hiopMatrix* Jac_d = kkt_->Jac_d_;
  const hiopMatrix* Hess = kkt_->Hess_;

  assert(kkt_->get_perturb_calc());
  hiopVector* delta_wx = kkt_->delta_wx_;
  hiopVector* delta_wd = kkt_->delta_wd_;
  hiopVector* delta_cc = kkt_->delta_cc_;
  hiopVector* delta_cd = kkt_->delta_cd_;
  if(kkt_->get_perturb_calc()) {
    kkt_->get_perturb_calc()->get_curr_perturbations(*delta_wx, *delta_wd, *delta_cc, *delta_cd);
  } else {
    delta_wx->setToZero();
    delta_wd->setToZero();
    delta_cc->setToZero();
    delta_cd->setToZero();
  }

  bool bret = split_vec_to_build_it(x);

  //rx = H*dx + delta_wx_*I*dx + Jc'*dyc + Jd'*dyd - dzl + dzu
  Hess->timesVec(0.0, *yrx_, +1.0, *dx_);
  yrx_->axzpy(1., *delta_wx, *dx_);
  Jac_c->transTimesVec(1.0, *yrx_, 1.0, *dyc_);
  Jac_d->transTimesVec(1.0, *yrx_, 1.0, *dyd_);
  yrx_->axpy(-1.0, *dzl_);
  yrx_->axpy( 1.0, *dzu_);

  //RD = delta_wd_*dd - dyd - dvl + dvu
  yrd_->setToZero();
  yrd_->axpy(-1., *dyd_);
  yrd_->axpy(-1., *dvl_);
  yrd_->axpy(+1., *dvu_);
  yrd_->axzpy(1., *delta_wd, *dd_);

  //RYC = Jc*dx - delta_cc_*dyc
  Jac_c->timesVec(0.0, *yryc_, 1.0, *dx_);
  yryc_->axzpy(-1., *delta_cc, *dyc_);

  //RYD = Jd*dx - dd - delta_cd_*dyd
  Jac_d->timesVec(0.0, *yryd_, 1.0, *dx_);
  yryd_->axpy(-1.0, *dd_);
  yryd_->axzpy(-1., *delta_cd, *dyd_);

  //RXL = -dx + Sxl*dsxl
  yrsxl_->setToZero();
  yrsxl_->axpy(-1.0, *dx_);
  yrsxl_->axzpy(1.0,*iter_->get_sxl(), *dsxl_);
  yrsxl_->selectPattern(kkt_->nlp_->get_ixl());
  
  //RXU = dx + Sxu dsxu
  yrsxu_->copyFrom(*dx_);
  yrsxu_->axzpy(1.0,*iter_->get_sxu(), *dsxu_);
  yrsxu_->selectPattern(kkt_->nlp_->get_ixu());

  //RDL = -dd + Sdl dsdl
  yrsdl_->setToZero();
  yrsdl_->axpy( -1.0, *dd_);
  yrsdl_->axzpy(1.0,*iter_->get_sdl(), *dsdl_);
  yrsdl_->selectPattern(kkt_->nlp_->get_idl());

  //RDU = dd + Sdu dsdu
  yrsdu_->setToZero();
  yrsdu_->axpy( 1.0, *dd_);
  yrsdu_->axzpy(1.0,*iter_->get_sdu(), *dsdu_);
  yrsdu_->selectPattern(kkt_->nlp_->get_idu());

  // dzxl + Zxl dsxl
  yrzl_->copyFrom(*dzl_);
  yrzl_->axzpy(1.0,*iter_->get_zl(), *dsxl_);

  // dzxu + Zxu dsxu
  yrzu_->copyFrom(*dzu_);
  yrzu_->axzpy(1.0,*iter_->get_zu(), *dsxu_);

  // dzdl + Zdl dsdl
  yrvl_->copyFrom(*dvl_);
  yrvl_->axzpy(1.0,*iter_->get_vl(), *dsdl_);

  // dzdu + Zdu dsdu
  yrvu_->copyFrom(*dvu_);
  yrvu_->axzpy(1.0,*iter_->get_vu(), *dsdu_);

  bret = combine_res_to_build_vec(y);
  return true;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopPrecondKKTOpr
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
hiopPrecondKKTOpr::hiopPrecondKKTOpr(hiopKKTLinSys* kkt, 
                               const hiopIterate* iter,
                               const hiopResidual* resid,
                               const hiopIterate* dir)
  : kkt_(kkt),
    iter_(iter),
    resid_(nullptr),
    dir_(nullptr)
{
  resid_ = new hiopResidual(kkt_->nlp_);
  dir_ = new hiopIterate(kkt_->nlp_);

  dx_ = dir_->get_x();
  dx_->setToZero();
  dd_ = dir_->get_d();
  dd_->setToZero();
  dyc_ = dir_->get_yc();
  dyc_->setToZero();
  dyd_ = dir_->get_yd();
  dyd_->setToZero();
  dsxl_ = dir_->get_sxl();
  dsxl_->setToZero();
  dsxu_ = dir_->get_sxu();
  dsxu_->setToZero();
  dsdl_ = dir_->get_sdl();
  dsdl_->setToZero();
  dsdu_ = dir_->get_sdu();
  dsdu_->setToZero();
  dzl_ = dir_->get_zl();
  dzl_->setToZero();
  dzu_ = dir_->get_zu();
  dzu_->setToZero();
  dvl_ = dir_->get_vl();
  dvl_->setToZero();
  dvu_ = dir_->get_vu();
  dvu_->setToZero();

  yrx_ = resid_->get_rx();
  yrx_->setToZero();
  yrd_ = resid_->get_rd();
  yrd_->setToZero();
  yryc_ = resid_->get_ryc();
  yryc_->setToZero();
  yryd_ = resid_->get_ryd();
  yryd_->setToZero();
  yrsxl_ = resid_->get_rxl();
  yrsxl_->setToZero();
  yrsxu_ = resid_->get_rxu();
  yrsxu_->setToZero();
  yrsdl_ = resid_->get_rdl();
  yrsdl_->setToZero();
  yrsdu_ = resid_->get_rdu();
  yrsdu_->setToZero();
  yrzl_ = resid_->get_rszl();
  yrzl_->setToZero();
  yrzu_ = resid_->get_rszu();
  yrzu_->setToZero();
  yrvl_ = resid_->get_rsvl();
  yrvl_->setToZero();
  yrvu_ = resid_->get_rsvu();
  yrvu_->setToZero();
}

bool hiopPrecondKKTOpr::split_vec_to_build_res(const hiopVector& vec)
{
  size_type nx = dx_->get_size();
  size_type neq = dyc_->get_size();
  size_type nineq = dyd_->get_size();
  assert(vec.get_local_size() == nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq+nineq);

  yrx_->startingAtCopyFromStartingAt(0,   vec, 0);
  yrd_->startingAtCopyFromStartingAt(0,   vec, nx);
  yryc_->startingAtCopyFromStartingAt(0,  vec, nx+nineq);
  yryd_->startingAtCopyFromStartingAt(0,  vec, nx+nineq+neq);
  yrsxl_->startingAtCopyFromStartingAt(0, vec, nx+nineq+neq+nineq);
  yrsxu_->startingAtCopyFromStartingAt(0, vec, nx+nineq+neq+nineq+nx);
  yrsdl_->startingAtCopyFromStartingAt(0, vec, nx+nineq+neq+nineq+nx+nx);
  yrsdu_->startingAtCopyFromStartingAt(0, vec, nx+nineq+neq+nineq+nx+nx+nineq);
  yrzl_->startingAtCopyFromStartingAt(0,  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq);
  yrzu_->startingAtCopyFromStartingAt(0,  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx);
  yrvl_->startingAtCopyFromStartingAt(0,  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx);
  yrvu_->startingAtCopyFromStartingAt(0,  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq);
  return true;
}

bool hiopPrecondKKTOpr::combine_dir_to_build_vec(hiopVector& vec)
{
  size_type nx = dx_->get_size();
  size_type neq = dyc_->get_size();
  size_type nineq = dyd_->get_size();
  assert(vec.get_local_size() == nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq+nineq);

  dx_->copyToStarting(   vec, 0);
  dd_->copyToStarting(   vec, nx);
  dyc_->copyToStarting(  vec, nx+nineq);
  dyd_->copyToStarting(  vec, nx+nineq+neq);
  dsxl_->copyToStarting( vec, nx+nineq+neq+nineq);
  dsxu_->copyToStarting( vec, nx+nineq+neq+nineq+nx);
  dsdl_->copyToStarting( vec, nx+nineq+neq+nineq+nx+nx);
  dsdu_->copyToStarting( vec, nx+nineq+neq+nineq+nx+nx+nineq);
  dzl_->copyToStarting(  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq);
  dzu_->copyToStarting(  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx);
  dvl_->copyToStarting(  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx);
  dvu_->copyToStarting(  vec, nx+nineq+neq+nineq+nx+nx+nineq+nineq+nx+nx+nineq);
  return true;
}

bool hiopPrecondKKTOpr::times_vec(hiopVector& y, const hiopVector& x)
{
  bool bret;
  bret = split_vec_to_build_res(x);

  bret = kkt_->computeDirections(resid_, dir_); 

  bret = combine_dir_to_build_vec(y);
  return true;
}

bool hiopPrecondKKTOpr::trans_times_vec(hiopVector& y, const hiopVector& x)
{
  // compressed preconditioner is symmetric
  return times_vec(y,x);
}


hiopKKTLinSysNormalEquation::hiopKKTLinSysNormalEquation(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressed(nlp)
{
  rd_tilde_  = Dd_->alloc_clone();
  ryc_tilde_ = nlp->alloc_dual_eq_vec();
  ryd_tilde_ = Dd_->alloc_clone();
  Hx_ = Dx_->alloc_clone();
  Hd_ = Dd_->alloc_clone();
  x_wrk_ = Dx_->alloc_clone();
  d_wrk_ = Dd_->alloc_clone();
}

hiopKKTLinSysNormalEquation::~hiopKKTLinSysNormalEquation()
{
  delete rd_tilde_;
  delete ryc_tilde_;
  delete ryd_tilde_;
  delete Hx_;
  delete Hd_;
  delete x_wrk_;
  delete d_wrk_;
}

bool hiopKKTLinSysNormalEquation::update(const hiopIterate* iter,
                                         const hiopVector* grad_f,
                                         const hiopMatrix* Jac_c,
                                         const hiopMatrix* Jac_d,
                                         hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.start_linsolve();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();
  
  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = Hess;

  size_type nx  = Hess_->m();
  size_type neq = Jac_c_->m();
  size_type nineq = Jac_d_->m();
  assert(nx==Hess_->n());
  assert(nx==Jac_c_->n());
  assert(nx==Jac_d_->n());

  // compute barrier diagonals (these change only between outer optimiz iterations)
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  assert(true==Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();
  
  //factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}


bool hiopKKTLinSysNormalEquation::computeDirections(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual &r = *resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde = rx + Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl) - Sdu^{-1}*(rsvu-Vu*rdu)
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector &rl=*(dir->x);//temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    //rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern( 1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local()) {
    //ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector &ru=*(dir->x);//temporary working buffer
    ru.copyFrom(*r.rszu); ru.axzpy(-1.0,*iter_->zu, *r.rxu);
    //rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  //for rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
  rd_tilde_->copyFrom(*r.rd);
  if(nlp_->m_ineq_low()) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    //rd_tilde +=  Sdl^{-1}*(rsvl-Vl*rdl)
    rd_tilde_->axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    //rd_tilde += -Sdu^{-1}(rsvu-Vu*rdu)
    rd_tilde_->axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * [ ryc_tilde ] = [ Jc  0 ] [ H+Dx+delta_wx_     0       ]^{-1}  [ rx_tilde ] - [ ryc ] 
   * [ ryd_tilde ]   [ Jd -I ] [   0           Dd+delta_wd_ ]       [ rd_tilde ]   [ ryd ]
   */

  /***********************************************************************
   * TODO: now we assume H is empty or diagonal
   * hence we have 
   * [ ryc_tilde ] = [ Jc ] [H+Dx+delta_wx_]^{-1} [ rx_tilde ] - [ ryc ] 
   * [ ryd_tilde ]   [ Jd ] [H+Dx+delta_wx_]^{-1} [ rx_tilde ] - [ Dd+delta_wd_ ]^{-1} [ rd_tilde ] - [ ryd ]
   */
  {
    /* x_wrk_ = [H+Dx+delta_wx_]^{-1} [ rx_tilde ] */
    x_wrk_->copyFrom(*rx_tilde_);
    x_wrk_->componentDiv(*Hx_);

    ryc_tilde_->copyFrom(*r.ryc);
    Jac_c_->timesVec(-1.0, *ryc_tilde_, 1.0, *x_wrk_);
    
    /* d_wrk_ = [ Dd+delta_wd_ ]^{-1} [ rd_tilde ] */
    d_wrk_->copyFrom(*rd_tilde_);
    d_wrk_->componentDiv(*Hd_);

    ryd_tilde_->copyFrom(*r.ryd);
    Jac_d_->timesVec(-1.0, *ryd_tilde_, 1.0, *x_wrk_);
    ryd_tilde_->axpy(-1.0, *d_wrk_);
  }

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*ryc_tilde_, *ryd_tilde_, *dir->yc, *dir->yd);

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  /***********************************************************************
  * TODO: now we assume H is empty or diagonal
  * hence from
  *   [ H+Dx+delta_wx_     0       ] [dx] = [ rx_tilde ] - [ Jc^T  Jd^T] [dyc]
  *   [   0           Dd+delta_wd_ ] [dd]   [ rd_tilde ]   [  0     -I ] [dyd]
  * we can recover
  *   [dx] = [ H+Dx+delta_wx_ ]^{-1} ( [ rx_tilde ] - [ Jc^T ] [dyc] - [Jd^T] [dyd] )
  *   [dd] = [ Dd+delta_wd_ ]^{-1}   ( [ rd_tilde ] + [dyd] ) 
  */
  dir->x->copyFrom(*rx_tilde_);
  Jac_c_->transTimesVec(1.0, *dir->x, -1.0, *dir->yc);
  Jac_d_->transTimesVec(1.0, *dir->x, -1.0, *dir->yd);
  dir->x->componentDiv(*Hx_);

  dir->d->copyFrom(*rd_tilde_);
  dir->d->axpy(1.0,*dir->yd);
  dir->d->componentDiv(*Hd_);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
  if(false == sol_ok) {
    nlp_->runStats.tmSolverInternal.stop();
    nlp_->runStats.linsolv.end_linsolve();
    return false;
  }
  
  bool bret = compute_directions_for_full_space(resid, dir);
  
  nlp_->runStats.tmSolverInternal.stop();
  nlp_->runStats.linsolv.end_linsolve();
  return true;
}


bool hiopKKTLinSysFull::test_direction(const hiopIterate* dir, hiopMatrix* Hess)
{
  bool retval;
  nlp_->runStats.tmSolverInternal.start();

  if(!x_wrk_) {
    x_wrk_ = nlp_->alloc_primal_vec();
    x_wrk_->setToZero();
  }
  if(!d_wrk_) {
    d_wrk_ = nlp_->alloc_dual_ineq_vec();
    d_wrk_->setToZero();
  }

  hiopVector* sol_x = dir->get_x();
  hiopVector* sol_d = dir->get_d();
  double dWd = 0;
  double xs_nrmsq = 0.0;
  double dbl_wrk;

  perturb_calc_->get_curr_perturbations(*delta_wx_, *delta_wd_, *delta_cc_, *delta_cd_);  

  /* compute xWx = x(H+Dx_)x (for primal var [x,d] */
  Hess_->timesVec(0.0, *x_wrk_, 1.0, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);
  
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  x_wrk_->setToZero();
  x_wrk_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  x_wrk_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  x_wrk_->componentMult(*sol_x);
  x_wrk_->axzpy(1., *delta_wx_, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  d_wrk_->setToZero();
  d_wrk_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  d_wrk_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  d_wrk_->componentMult(*sol_d);
  d_wrk_->axzpy(1., *delta_wd_, *sol_d);
  dWd += d_wrk_->dotProductWith(*sol_d);

  /* compute rhs for the dWd test */
  dbl_wrk = sol_x->twonorm();
  xs_nrmsq += dbl_wrk*dbl_wrk;
  dbl_wrk = sol_d->twonorm();
  xs_nrmsq += dbl_wrk*dbl_wrk;

  if(dWd < xs_nrmsq * nlp_->options->GetNumeric("neg_curv_test_fact")) {
    // have negative curvature. Add regularization and re-factorize the matrix
    retval = false;
  } else {
    // have positive curvature. Accept this factoraizaiton and direction.
    retval = true;
  }
  
  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}



};

