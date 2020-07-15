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

#include "hiopDualsUpdater.hpp"
#include "hiopFactory.hpp"

#include "hiop_blasdefs.hpp"

namespace hiop
{

hiopDualsLsqUpdate::hiopDualsLsqUpdate(hiopNlpFormulation* nlp) 
  : hiopDualsUpdater(nlp) 
{
  hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(_nlp);
  assert(NULL!=nlpd);
  _mexme = getMatrixDenseInstance(nlpd->m_eq(),   nlpd->m_eq());
  _mexmi = getMatrixDenseInstance(nlpd->m_eq(),   nlpd->m_ineq());
  _mixmi = getMatrixDenseInstance(nlpd->m_ineq(), nlpd->m_ineq());
  _mxm   = getMatrixDenseInstance(nlpd->m(), nlpd->m());

  M      = getMatrixDenseInstance(nlpd->m(), nlpd->m());
  rhs    = getVectorInstance(nlpd->m());
  rhsc   = dynamic_cast<hiopVectorPar*>(nlpd->alloc_dual_eq_vec());
  rhsc->setToZero();
  rhsd   = dynamic_cast<hiopVectorPar*>(nlpd->alloc_dual_ineq_vec());
  rhsd->setToZero();
  _vec_n = dynamic_cast<hiopVectorPar*>(nlpd->alloc_primal_vec());
  _vec_mi= dynamic_cast<hiopVectorPar*>(nlpd->alloc_dual_ineq_vec());
#ifdef HIOP_DEEPCHECKS
  M_copy = M->alloc_clone();
  rhs_copy = rhs->alloc_clone();
  _mixme = getMatrixDenseInstance(nlpd->m_ineq(), nlpd->m_eq());
#endif
  //user options
  recalc_lsq_duals_tol = 1e-6;
};

hiopDualsLsqUpdate::~hiopDualsLsqUpdate()
{
  delete _mexme;
  delete _mexmi;
  delete _mixmi;
  delete _mxm;
  delete M;
  delete rhs;
  delete rhsc; 
  delete rhsd;
  delete _vec_n;
  delete _vec_mi;
#ifdef HIOP_DEEPCHECKS
  delete M_copy;
  delete rhs_copy;
  delete _mixme;
#endif
}


bool hiopDualsLsqUpdate::
go(const hiopIterate& iter,  hiopIterate& iter_plus,
   const double& f, const hiopVector& c, const hiopVector& d,
   const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
   const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual,
   const double& mu, const double& kappa_sigma, const double& infeas_nrm_trial)
{
  hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(_nlp);
  assert(nlpd!=NULL);

  //first update the duals using steplength along the search directions. This is fine for 
  //signed duals z_l, z_u, v_l, and v_u. The rest of the duals, yc and yd, will be found as a 
  //solution to the above LSQ problem
  if(!iter_plus.takeStep_duals(iter, search_dir, alpha_primal, alpha_dual)) {
    nlpd->log->printf(hovError, "dual lsq update: error in standard update of the duals");
    return false;
  }
  if(!iter_plus.adjustDuals_primalLogHessian(mu,kappa_sigma)) {
    nlpd->log->printf(hovError, "dual lsq update: error in adjustDuals");
    return false;
  }


  //return if the constraint violation (primal infeasibility) is not below the tol for the LSQ update
  if(infeas_nrm_trial>recalc_lsq_duals_tol) {
    nlpd->log->printf(hovScalars, "will not perform the dual lsq update since the primal infeasibility (%g) "
		      "is not under the tolerance recalc_lsq_duals_tol=%g.\n",
		      infeas_nrm_trial, recalc_lsq_duals_tol);
    return true;
  }

  return LSQUpdate(iter_plus, grad_f, jac_c, jac_d);
};

/** Given xk, zk_l, zk_u, vk_l, and vk_u (contained in 'iter'), this method solves an LSQ problem 
 * corresponding to dual infeasibility equation
 *    min_{y_c,y_d} ||  \nabla f(xk) + J^T_c(xk) y_c + J_d^T(xk) y_d - zk_l+zk_u  ||^2
 *                  || - y_d - vk_l + vk_u                                        ||_2,
 *  which is
 *   min_{y_c, y_d} || [ J_c^T  J_d^T ] [ y_c ]  -  [ -\nabla f(xk) + zk_l-zk_u ]  ||^2
 *                  || [  0       I   ] [ y_d ]     [ - vk_l + vk_u               ]  ||_2
 *  We compute y_c and y_d by solving the linear system
 *   [ J_c J_c^T    J_c J_d^T     ] [ y_c ]  =  [ J_c   0 ] [ -\nabla f(xk) + zk_l-zk_u ] 
 *   [ J_d J_c^T    J_d J_d^T + I ] [ y_d ]     [ J_d   I ] [ - vk_l + vk_u              ]
 *
 * This linear system is small (of size m=m_E+m_I) (so it is replicated for all MPI ranks).
 * 
 * The matrix of the above system is stored in the member variable M of this class and the
 *  right-hand side in 'rhs'
 */
bool hiopDualsLsqUpdate::
LSQUpdate(hiopIterate& iter, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d)
{
  hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(_nlp);
  assert(nlpd!=NULL);

  //compute terms in M: Jc * Jc^T, J_c * J_d^T, and J_d * J_d^T
  //! streamline the communication (use _mxm as a global buffer for the MPI_Allreduce)
  jac_c.timesMatTrans(0.0, *_mexme, 1.0, jac_c);
  jac_c.timesMatTrans(0.0, *_mexmi, 1.0, jac_d);
  jac_d.timesMatTrans(0.0, *_mixmi, 1.0, jac_d);
  _mixmi->addDiagonal(1.0);

  M->copyBlockFromMatrix(0,0,*_mexme);
  M->copyBlockFromMatrix(0, nlpd->m_eq(), *_mexmi);
  M->copyBlockFromMatrix(nlpd->m_eq(),nlpd->m_eq(), *_mixmi);

#ifdef HIOP_DEEPCHECKS
  M_copy->copyFrom(*M);
  jac_d.timesMatTrans(0.0, *_mixme, 1.0, jac_c);
  M_copy->copyBlockFromMatrix(nlpd->m_eq(), 0, *_mixme);
  M_copy->assertSymmetry(1e-12);
#endif

  //bailout in case there is an error in the Cholesky factorization
  int info;
  if((info=this->factorizeMat(*M))) {
    nlpd->log->printf(hovError, "dual lsq update: error %d in the Cholesky factorization.\n", info);
    return false;
  }

  // compute rhs=[rhsc,rhsd]. 
  // [ rhsc ] = - [ J_c   0 ] [ vecx ] 
  // [ rhsd ]     [ J_d   I ] [ vecd ]
  // [vecx,vecd] = - [ -\nabla f(xk) + zk_l-zk_u, - vk_l + vk_u]. 
  hiopVector& vecx = *_vec_n;
  vecx.copyFrom(grad_f);
  vecx.axpy(-1.0, *iter.get_zl());
  vecx.axpy( 1.0, *iter.get_zu());

  hiopVector& vecd = *_vec_mi;
  vecd.copyFrom(*iter.get_vl());
  vecd.axpy(-1.0, *iter.get_vu());

  jac_c.timesVec(0.0, *rhsc, -1.0, vecx);
  jac_d.timesVec(0.0, *rhsd, -1.0, vecx);
  rhsd->axpy(-1.0, vecd);

  rhs->copyFromStarting(0, *rhsc);
  rhs->copyFromStarting(nlpd->m_eq(), *rhsd);

  //nlpd->log->write("rhs", *rhs, hovSummary);
#ifdef HIOP_DEEPCHECKS
  rhs_copy->copyFrom(*rhs);
#endif

  //solve for this rhs
  if((info=this->solveWithFactors(*M, *rhs))) {
    nlpd->log->printf(hovError, "dual lsq update: error %d in the solution process.\n", info);
    return false;
  }

  //update yc and yd in iter_plus
  rhs->copyToStarting(0, *iter.get_yc());
  rhs->copyToStarting(nlpd->m_eq(), *iter.get_yd());

#ifdef HIOP_DEEPCHECKS
  double nrmrhs = rhs_copy->twonorm();
  M_copy->timesVec(-1.0,  *rhs_copy, 1.0, *rhs);
  double nrmres = rhs_copy->twonorm() / (1+nrmrhs);
  if(nrmres>1e-4) {
    nlpd->log->printf(hovError,
		      "hiopDualsLsqUpdate::LSQUpdate linear system residual is dangerously high: %g\n", nrmres);
    assert(false && "hiopDualsLsqUpdate::LSQUpdate linear system residual is dangerously high");
    return false;
  } else {
    if(nrmres>1e-6)
      nlpd->log->printf(hovWarning,
			"hiopDualsLsqUpdate::LSQUpdate linear system residual is dangerously high: %g\n", nrmres);
  }
#endif

  //nlpd->log->write("yc ini", *iter.get_yc(), hovSummary);
  //nlpd->log->write("yd ini", *iter.get_yd(), hovSummary);
  return true;
};

int hiopDualsLsqUpdate::factorizeMat(hiopMatrixDense& M)
{
#ifdef HIOP_DEEPCHECKS
  assert(M.m()==M.n());
#endif
  if(M.m()==0) return 0;
  char uplo='L'; int N=M.n(), lda=N, info;
  DPOTRF(&uplo, &N, M.local_buffer(), &lda, &info);
  if(info>0)
    _nlp->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf (Chol fact) detected %d minor being indefinite.\n", info);
  else
    if(info<0) 
      _nlp->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf returned error %d\n", info);
  assert(info==0);
  return info;
}

int hiopDualsLsqUpdate::solveWithFactors(hiopMatrixDense& M, hiopVector& r)
{
#ifdef HIOP_DEEPCHECKS
  assert(M.m()==M.n());
#endif
  if(M.m()==0) return 0;
  char uplo='L'; //we have upper triangular in C++, but this is lower in fortran
  int N=M.n(), lda=N, nrhs=1, info;
  DPOTRS(&uplo,&N, &nrhs, M.local_buffer(), &lda, r.local_data(), &lda, &info);
  if(info<0) 
    _nlp->log->printf(hovError, "hiopKKTLinSysLowRank::solveWithFactors: dpotrs returned error %d\n", info);
#ifdef HIOP_DEEPCHECKS
  assert(info<=0);
#endif
  return info;
}


}; //~ end of namespace
