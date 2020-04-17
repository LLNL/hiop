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
#include "hiop_blasdefs.hpp"

#include <cmath>

namespace hiop
{

#ifdef HIOP_DEEPCHECKS
//computes the solve error for the KKT Linear system; used only for correctness checking
double hiopKKTLinSys::errorKKT(const hiopResidual* resid, const hiopIterate* sol)
{

  nlp->log->printf(hovLinAlgScalars, "KKT LinSys::errorKKT KKT_large residuals norm:\n");
  double derr=1e20,aux;
  hiopVectorPar *RX=resid->rx->new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd +dzl-dzu = rx
  //Hess->timesVec_noLogBarrierTerm(1.0, *RX, -1.0, *sol->x);
  HessianTimesVec_noLogBarrierTerm(1.0, *RX, -1.0, *sol->x);

  Jac_c->transTimesVec(1.0, *RX, -1.0, *sol->yc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, *sol->yd);
  //sol->zl->print("zl");
  //sol->zu->print("zu");
  RX->axpy( 1.0,*sol->zl);
  RX->axpy(-1.0,*sol->zu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rx=%g\n", aux);

  hiopVectorPar* RYC=resid->ryc->new_copy();
  Jac_c->timesVec(1.0, *RYC, -1.0, *sol->x);
  aux=RYC->infnorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- ryc=%g\n", aux);
  delete RYC;

  //RYD=ryd-Jd*dx+dd
  hiopVectorPar* RYD=resid->ryd->new_copy();
  Jac_d->timesVec(1.0, *RYD, -1.0, *sol->x);
  RYD->axpy(1.0,*sol->d);
  aux=RYD->infnorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- ryd=%g\n", aux);
  delete RYD; 

  //RXL=rxl+x-sxl
  RX->copyFrom(*resid->rxl);
  RX->axpy( 1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxl);
  RX->selectPattern(nlp->get_ixl());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rxl=%g\n", aux);
  //RXU=rxu-x-sxu
  RX->copyFrom(*resid->rxu);
  RX->axpy(-1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxu);
  RX->selectPattern(nlp->get_ixu());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rxu=%g\n", aux);
 

  //RDL=rdl+d-sdl
  hiopVectorPar* RD=resid->rdl->new_copy();
  RD->axpy( 1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdl);
  RD->selectPattern(nlp->get_idl());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rdl=%g\n", aux);
  //RDU=rdu-d-sdu
  RD->copyFrom(*resid->rdu);
  RD->axpy(-1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdu);
  RD->selectPattern(nlp->get_idu());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rdu=%g\n", aux);

  
  //complementarity residuals checks: rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszl);
  RX->axzpy(-1.0,*iter->sxl,*sol->zl);
  RX->axzpy(-1.0,*iter->zl, *sol->sxl);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rszl=%g\n", aux);
  //rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszu);
  RX->axzpy(-1.0,*iter->sxu,*sol->zu);
  RX->axzpy(-1.0,*iter->zu, *sol->sxu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rszu=%g\n", aux);
  delete RX; RX=NULL;
 
  //complementarity residuals checks: rsvl - Sdl dvl - Vl dsdl
  RD->copyFrom(*resid->rsvl);
  RD->axzpy(-1.0,*iter->sdl,*sol->vl);
  RD->axzpy(-1.0,*iter->vl, *sol->sdl);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rsvl=%g\n", aux);
  //complementarity residuals checks: rsvu - Sdu dvu - Vu dsdu
  RD->copyFrom(*resid->rsvu);
  RD->axzpy(-1.0,*iter->sdu,*sol->vu);
  RD->axzpy(-1.0,*iter->vu, *sol->sdu);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  nlp->log->printf(hovLinAlgScalars, "  --- rsvu=%g\n", aux);

  delete RD; RD=NULL;
  return derr;
}

#endif

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressedXYcYd
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Provides the functionality for reducing the KKT linear system to the 
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
hiopKKTLinSysCompressedXYcYd::hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp_)
  : hiopKKTLinSysCompressed(nlp_)
{
  Dd_inv = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_ineq_vec());
  assert(Dd_inv != NULL);

  ryd_tilde = Dd_inv->alloc_clone(); 
}

hiopKKTLinSysCompressedXYcYd::~hiopKKTLinSysCompressedXYcYd()
{
  delete Dd_inv;  
  delete ryd_tilde;
}

bool hiopKKTLinSysCompressedXYcYd::computeDirections(const hiopResidual* resid, 
						     hiopIterate* dir)
{
  nlp->runStats.tmSolverInternal.start();
  const hiopResidual &r=*resid; 

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde  = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * ryd_tilde = ryd + [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}*
   *                     [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)]
   */
  rx_tilde->copyFrom(*r.rx); 
  if(nlp->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVectorPar &rl=*(dir->x);//temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter->zl, *r.rxl);
    //rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde->axdzpy_w_pattern( 1.0, rl, *iter->sxl, nlp->get_ixl());
  }
  if(nlp->n_upp_local()) {
    //ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVectorPar &ru=*(dir->x);//temporary working buffer
    ru.copyFrom(*r.rszu); ru.axzpy(-1.0,*iter->zu, *r.rxu);
    //rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde->axdzpy_w_pattern(-1.0, ru, *iter->sxu, nlp->get_ixu());
  }
  
  //for ryd_tilde: 
  ryd_tilde->copyFrom(*r.ryd);
  // 1. the diag (Sdl^{-1}Vl+Sdu^{-1}Vu)^{-1} has already computed in Dd_inv in 'update'
  // 2. compute the left multiplicand in ryd2 (using buffer dir->sdl), that is
  //   ryd2 = [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)] (this is \tilde{r}_d in the notes)
  //    Inner ops are performed by accumulating in rd2  (buffer dir->sdu)
  hiopVectorPar &ryd2=*dir->sdl; ryd2.copyFrom(*r.rd);
  if(nlp->m_ineq_low()) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl); 
    rd2.axzpy(-1.0, *iter->vl, *r.rdl);
    //ryd2 +=  Sdl^{-1}*(rsvl-Vl*rdl)
    ryd2.axdzpy_w_pattern(1.0, rd2, *iter->sdl, nlp->get_idl());
  }
  if(nlp->m_ineq_upp()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu); 
    rd2.axzpy(-1.0, *iter->vu, *r.rdu);
    //ryd2 += -Sdu^{-1}(rsvu-Vu*rdu)
    ryd2.axdzpy_w_pattern(-1.0, rd2, *iter->sdu, nlp->get_idu());
  }

  nlp->log->write("Dinv (in computeDirections)", *Dd_inv, hovMatrices);

  //now the final ryd_tilde += Dd^{-1}*ryd2
  ryd_tilde->axzpy(1.0, ryd2, *Dd_inv);

#ifdef HIOP_DEEPCHECKS
  hiopVectorPar* rx_tilde_save=rx_tilde->new_copy();
  hiopVectorPar* ryc_save=r.ryc->new_copy();
  hiopVectorPar* ryd_tilde_save=ryd_tilde->new_copy();
#endif


  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function) 
   ***********************************************************************/
  solveCompressed(*rx_tilde, *r.ryc, *ryd_tilde, *dir->x, *dir->yc, *dir->yd);
  //recover dir->d = (D)^{-1}*(dir->yd + ryd2)
  dir->d->copyFrom(ryd2);
  dir->d->axpy(1.0,*dir->yd);
  dir->d->componentMult(*Dd_inv);

  //dir->d->print();

#ifdef HIOP_DEEPCHECKS
  errorCompressedLinsys(*rx_tilde_save,*ryc_save,*ryd_tilde_save, *dir->x, *dir->yc, *dir->yd);
  delete rx_tilde_save;
  delete ryc_save;
  delete ryd_tilde_save;
#endif

  /***********************************************************************
   * compute the rest of the directions
   *
   */
  //dsxl = rxl + dx  and dzl= [Sxl]^{-1} ( - Zl*dsxl + rszl)
  if(nlp->n_low_local()) { 
    dir->sxl->copyFrom(*r.rxl); dir->sxl->axpy( 1.0,*dir->x); dir->sxl->selectPattern(nlp->get_ixl()); 

    dir->zl->copyFrom(*r.rszl); dir->zl->axzpy(-1.0,*iter->zl,*dir->sxl); 
    dir->zl->componentDiv_p_selectPattern(*iter->sxl, nlp->get_ixl());
  } else {
    dir->sxl->setToZero(); dir->zl->setToZero();
  }

  //dir->sxl->print();
  //dir->zl->print();
  //dsxu = rxu - dx and dzu = [Sxu]^{-1} ( - Zu*dsxu + rszu)
  if(nlp->n_upp_local()) { 
    dir->sxu->copyFrom(*r.rxu); dir->sxu->axpy(-1.0,*dir->x); dir->sxu->selectPattern(nlp->get_ixu()); 

    dir->zu->copyFrom(*r.rszu); dir->zu->axzpy(-1.0,*iter->zu,*dir->sxu); dir->zu->selectPattern(nlp->get_ixu());
    dir->zu->componentDiv_p_selectPattern(*iter->sxu, nlp->get_ixu());
  } else {
    dir->sxu->setToZero(); dir->zu->setToZero();
  }

  //dir->sxu->print();
  //dir->zu->print();
  //dsdl = rdl + dd and dvl = [Sdl]^{-1} ( - Vl*dsdl + rsvl)
  if(nlp->m_ineq_low()) {
    dir->sdl->copyFrom(*r.rdl); dir->sdl->axpy( 1.0,*dir->d); dir->sdl->selectPattern(nlp->get_idl());

    dir->vl->copyFrom(*r.rsvl); dir->vl->axzpy(-1.0,*iter->vl,*dir->sdl); dir->vl->selectPattern(nlp->get_idl());
    dir->vl->componentDiv_p_selectPattern(*iter->sdl, nlp->get_idl());
  } else {
    dir->sdl->setToZero(); dir->vl->setToZero();
  }

  //dir->sdl->print();
  // dir->vl->print();
  //dsdu = rdu - dd and dvu = [Sdu]^{-1} ( - Vu*dsdu + rsvu )
  if(nlp->m_ineq_upp()>0) {
    dir->sdu->copyFrom(*r.rdu); dir->sdu->axpy(-1.0,*dir->d); dir->sdu->selectPattern(nlp->get_idu());
    
    dir->vu->copyFrom(*r.rsvu); dir->vu->axzpy(-1.0,*iter->vu,*dir->sdu); dir->vu->selectPattern(nlp->get_idu());
    dir->vu->componentDiv_p_selectPattern(*iter->sdu, nlp->get_idu());
  } else {
    dir->sdu->setToZero(); dir->vu->setToZero();
  }

  //dir->sdu->print();
  //dir->vu->print();
#ifdef HIOP_DEEPCHECKS
  assert(dir->sxl->matchesPattern(nlp->get_ixl()));
  assert(dir->sxu->matchesPattern(nlp->get_ixu()));
  assert(dir->sdl->matchesPattern(nlp->get_idl()));
  assert(dir->sdu->matchesPattern(nlp->get_idu()));
  assert(dir->zl->matchesPattern(nlp->get_ixl()));
  assert(dir->zu->matchesPattern(nlp->get_ixu()));
  assert(dir->vl->matchesPattern(nlp->get_idl()));
  assert(dir->vu->matchesPattern(nlp->get_idu()));

  //CHECK THE SOLUTION
  errorKKT(resid,dir);
#endif
  nlp->runStats.tmSolverInternal.stop();
  return true;
}

#ifdef HIOP_DEEPCHECKS
double hiopKKTLinSysCompressedXYcYd::
errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
		      const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd)
{
  nlp->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXYcYd::errorCompressedLinsys residuals norm:\n");
  
  double derr=1e20, aux;
  hiopVectorPar *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx, dx);
  
  Jac_c->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX; RX=NULL;
  
  hiopVectorPar* RC=ryc.new_copy();
  Jac_c->timesVec(1.0, *RC, -1.0, dx);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC; RC=NULL;
  
  hiopVectorPar* RD=ryd.new_copy();
  Jac_d->timesVec(1.0, *RD, -1.0, dx);
  RD->axzpy(1.0, *Dd_inv, dyd);
  aux = RD->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RD; RD=NULL;
  
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
hiopKKTLinSysCompressedXDYcYd::hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp_)
  : hiopKKTLinSysCompressed(nlp_)
{
  Dd = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_ineq_vec());
  assert(Dd != NULL);

  rd_tilde = Dd->alloc_clone(); 
}

hiopKKTLinSysCompressedXDYcYd::~hiopKKTLinSysCompressedXDYcYd()
{
  delete Dd;  
  delete rd_tilde;
}

bool hiopKKTLinSysCompressedXDYcYd::computeDirections(const hiopResidual* resid, 
						      hiopIterate* dir)
{
  nlp->runStats.tmSolverInternal.start();
  const hiopResidual &r=*resid; 

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
   *
   * rd_tilde = ryd + [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}*
   *                     [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)]
   */
  rx_tilde->copyFrom(*r.rx); 
  if(nlp->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVectorPar &rl=*(dir->x);//temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter->zl, *r.rxl);
    //rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde->axdzpy_w_pattern( 1.0, rl, *iter->sxl, nlp->get_ixl());
  }
  if(nlp->n_upp_local()) {
    //ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVectorPar &ru=*(dir->x);//temporary working buffer
    ru.copyFrom(*r.rszu); ru.axzpy(-1.0,*iter->zu, *r.rxu);
    //rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde->axdzpy_w_pattern(-1.0, ru, *iter->sxu, nlp->get_ixu());
  }
  
  //for rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
  rd_tilde->copyFrom(*r.rd);
  if(nlp->m_ineq_low()) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl); 
    rd2.axzpy(-1.0, *iter->vl, *r.rdl);
    //rd_tilde +=  Sdl^{-1}*(rsvl-Vl*rdl)
    rd_tilde->axdzpy_w_pattern(1.0, rd2, *iter->sdl, nlp->get_idl());
  }
  if(nlp->m_ineq_upp()>0) {
    hiopVector& rd2=*dir->sdu;
    //rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu); 
    rd2.axzpy(-1.0, *iter->vu, *r.rdu);
    //rd_tilde += -Sdu^{-1}(rsvu-Vu*rdu)
    rd_tilde->axdzpy_w_pattern(-1.0, rd2, *iter->sdu, nlp->get_idu());
  }
  nlp->log->write("Dd (in computeDirections)", *Dd, hovMatrices);

#ifdef HIOP_DEEPCHECKS
  hiopVectorPar* rx_tilde_save = rx_tilde->new_copy();
  hiopVectorPar* rd_tilde_save = rd_tilde->new_copy();
  hiopVectorPar* ryc_save = r.ryc->new_copy();
  hiopVectorPar* ryd_save = r.ryd->new_copy();
#endif


  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function) 
   ***********************************************************************/
  solveCompressed(*rx_tilde, *rd_tilde, *r.ryc, *r.ryd, *dir->x, *dir->d, *dir->yc, *dir->yd);

#ifdef HIOP_DEEPCHECKS
  double derr = 
    errorCompressedLinsys(*rx_tilde_save, *rd_tilde_save, *ryc_save, *ryd_save, 
			  *dir->x, *dir->d, *dir->yc, *dir->yd);
  if(derr>1e-8)
    nlp->log->printf(hovWarning, "solve compressed high absolute resid norm (=%12.5e)\n", derr);
  delete rx_tilde_save;
  delete ryc_save;
  delete rd_tilde_save;
  delete ryd_save;
#endif

  /***********************************************************************
   * compute the rest of the directions
   *
   */
  //dsxl = rxl + dx  and dzl= [Sxl]^{-1} ( - Zl*dsxl + rszl)
  if(nlp->n_low_local()) { 
    dir->sxl->copyFrom(*r.rxl); dir->sxl->axpy( 1.0,*dir->x); dir->sxl->selectPattern(nlp->get_ixl()); 

    dir->zl->copyFrom(*r.rszl); dir->zl->axzpy(-1.0,*iter->zl,*dir->sxl); 
    dir->zl->componentDiv_p_selectPattern(*iter->sxl, nlp->get_ixl());
  } else {
    dir->sxl->setToZero(); dir->zl->setToZero();
  }

  //dir->sxl->print();
  //dir->zl->print();
  //dsxu = rxu - dx and dzu = [Sxu]^{-1} ( - Zu*dsxu + rszu)
  if(nlp->n_upp_local()) { 
    dir->sxu->copyFrom(*r.rxu); dir->sxu->axpy(-1.0,*dir->x); dir->sxu->selectPattern(nlp->get_ixu()); 

    dir->zu->copyFrom(*r.rszu); dir->zu->axzpy(-1.0,*iter->zu,*dir->sxu); dir->zu->selectPattern(nlp->get_ixu());
    dir->zu->componentDiv_p_selectPattern(*iter->sxu, nlp->get_ixu());
  } else {
    dir->sxu->setToZero(); dir->zu->setToZero();
  }

  //dir->sxu->print();
  //dir->zu->print();
  //dsdl = rdl + dd and dvl = [Sdl]^{-1} ( - Vl*dsdl + rsvl)
  if(nlp->m_ineq_low()) {
    dir->sdl->copyFrom(*r.rdl); dir->sdl->axpy( 1.0,*dir->d); dir->sdl->selectPattern(nlp->get_idl());

    dir->vl->copyFrom(*r.rsvl); dir->vl->axzpy(-1.0,*iter->vl,*dir->sdl); dir->vl->selectPattern(nlp->get_idl());
    dir->vl->componentDiv_p_selectPattern(*iter->sdl, nlp->get_idl());
  } else {
    dir->sdl->setToZero(); dir->vl->setToZero();
  }

  //dsdu = rdu - dd and dvu = [Sdu]^{-1} ( - Vu*dsdu + rsvu )
  if(nlp->m_ineq_upp()>0) {
    dir->sdu->copyFrom(*r.rdu); dir->sdu->axpy(-1.0,*dir->d); dir->sdu->selectPattern(nlp->get_idu());
    
    dir->vu->copyFrom(*r.rsvu); dir->vu->axzpy(-1.0,*iter->vu,*dir->sdu); dir->vu->selectPattern(nlp->get_idu());
    dir->vu->componentDiv_p_selectPattern(*iter->sdu, nlp->get_idu());
  } else {
    dir->sdu->setToZero(); dir->vu->setToZero();
  }

#ifdef HIOP_DEEPCHECKS
  assert(dir->sxl->matchesPattern(nlp->get_ixl()));
  assert(dir->sxu->matchesPattern(nlp->get_ixu()));
  assert(dir->sdl->matchesPattern(nlp->get_idl()));
  assert(dir->sdu->matchesPattern(nlp->get_idu()));
  assert(dir->zl->matchesPattern(nlp->get_ixl()));
  assert(dir->zu->matchesPattern(nlp->get_ixu()));
  assert(dir->vl->matchesPattern(nlp->get_idl()));
  assert(dir->vu->matchesPattern(nlp->get_idu()));

  //CHECK THE SOLUTION
  errorKKT(resid,dir);
#endif
  nlp->runStats.tmSolverInternal.stop();
  return true;
}

#ifdef HIOP_DEEPCHECKS
double hiopKKTLinSysCompressedXDYcYd::
errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& rd, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
		      const hiopVectorPar& dx, const hiopVectorPar& dd, const hiopVectorPar& dyc, const hiopVectorPar& dyd)
{
  nlp->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXDYcYd::errorCompressedLinsys residuals norm:\n");
  
  double derr=-1., aux;
  hiopVectorPar *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx, dx);

  Jac_c->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX; RX=NULL;

  //RD = rd + dyd - Dd*dd
  hiopVectorPar* RD=rd.new_copy();
  RD->axpy( 1., dyd);
  RD->axzpy(-1., *Dd, dd);
  aux=RD->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >>  rd=%g\n", aux);
  delete RD; RD=NULL;

  hiopVectorPar* RC=ryc.new_copy();
  Jac_c->timesVec(1.0, *RC, -1.0, dx);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC; RC=NULL;
  
  //RYD = ryd+dyd - Jd*dx
  hiopVectorPar* RYD=ryd.new_copy();
  Jac_d->timesVec(1.0, *RYD, -1.0, dx);
  RYD->axpy(1.0, dd);
  aux = RYD->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RYD; RYD=NULL;
  
  return derr;
}
#endif


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysLowRank
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

hiopKKTLinSysLowRank::hiopKKTLinSysLowRank(hiopNlpFormulation* nlp_)
  : hiopKKTLinSysCompressedXYcYd(nlp_)
{
  nlpD = dynamic_cast<hiopNlpDenseConstraints*>(nlp_);

  _kxn_mat = nlpD->alloc_multivector_primal(nlpD->m()); //!opt
  N = new hiopMatrixDense(nlpD->m(),nlpD->m());
#ifdef HIOP_DEEPCHECKS
  Nmat=N->alloc_clone();
#endif
  _k_vec1 = dynamic_cast<hiopVectorPar*>(nlpD->alloc_dual_vec());
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
update(const hiopIterate* iter_, 
       const hiopVector* grad_f_, 
       const hiopMatrixDense* Jac_c_, const hiopMatrixDense* Jac_d_, 
       hiopHessianLowRank* Hess_)
{
  nlp->runStats.tmSolverInternal.start();

  iter=iter_;
  grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_);
  Jac_c = Jac_c_; Jac_d = Jac_d_;
  //Hess = dynamic_cast<hiopHessianInvLowRank*>(Hess_);
  Hess=HessLowRank=Hess_;

  //compute the diagonals
  //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx->setToZero();
  Dx->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp->get_ixl());
  Dx->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp->get_ixu());
  nlp->log->write("Dx in KKT", *Dx, hovMatrices);

  HessLowRank->updateLogBarrierDiagonal(*Dx);

  //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_inv->setToZero();
  Dd_inv->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
  Dd_inv->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef HIOP_DEEPCHECKS
  assert(true==Dd_inv->allPositive());
#endif 
  Dd_inv->invert();

  nlp->runStats.tmSolverInternal.stop();

  nlp->log->write("Dd_inv in KKT", *Dd_inv, hovMatrices);
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
void hiopKKTLinSysLowRank::
solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
		hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
{
#ifdef HIOP_DEEPCHECKS
  //some outputing
  nlp->log->write("KKT Low rank: solve compressed RHS", hovIteration);
  nlp->log->write("  rx: ",  rx, hovIteration); nlp->log->write(" ryc: ", ryc, hovIteration); nlp->log->write(" ryd: ", ryd, hovIteration);
  nlp->log->write("  Jc: ", *Jac_c, hovMatrices);
  nlp->log->write("  Jd: ", *Jac_d, hovMatrices);
  nlp->log->write("  Dd_inv: ", *Dd_inv, hovMatrices);
  assert(Dd_inv->isfinite() && "Something bad happened: nan or inf value");
#endif

  hiopMatrixDense& J = *_kxn_mat;
  const hiopMatrixDense* Jac_c_de = dynamic_cast<const hiopMatrixDense*>(Jac_c); assert(Jac_c_de);
  const hiopMatrixDense* Jac_d_de = dynamic_cast<const hiopMatrixDense*>(Jac_d); assert(Jac_d_de);
  J.copyRowsFrom(*Jac_c_de, nlp->m_eq(), 0); //!opt
  J.copyRowsFrom(*Jac_d_de, nlp->m_ineq(), nlp->m_eq());//!opt

  //N =  J*(Hess\J')
  //Hess->symmetricTimesMat(0.0, *N, 1.0, J);
  HessLowRank->symMatTimesInverseTimesMatTrans(0.0, *N, 1.0, J);

  //subdiag of N += 1., Dd_inv
  N->addSubDiagonal(1., nlp->m_eq(), *Dd_inv);
#ifdef HIOP_DEEPCHECKS
  assert(J.isfinite());
  nlp->log->write("solveCompressed: N is", *N, hovMatrices);
  nlp->log->write("solveCompressed: rx is", rx, hovMatrices);
  nlp->log->printf(hovLinAlgScalars, "inf norm of Dd_inv is %g\n", Dd_inv->infnorm());
  N->assertSymmetry(1e-10);
#endif
 
  //compute the rhs of the lin sys involving N 
  //  1. first compute (H+Dx)^{-1} rx_tilde and store it temporarily in dx
  HessLowRank->solve(rx, dx);
#ifdef HIOP_DEEPCHECKS
  assert(rx.isfinite() && "Something bad happened: nan or inf value");
  assert(dx.isfinite() && "Something bad happened: nan or inf value");
#endif
  
  // 2 . then rhs =   [ Jc(H+Dx)^{-1}*rx - ryc ]
  //                  [ Jd(H+dx)^{-1}*rx - ryd ]
  hiopVectorPar& rhs=*_k_vec1;
  rhs.copyFromStarting(0, ryc);
  rhs.copyFromStarting(nlp->m_eq(), ryd);
  J.timesVec(-1.0, rhs, 1.0, dx);

#ifdef HIOP_DEEPCHECKS
  nlp->log->write("solveCompressed: dx sol is", dx, hovMatrices);
  nlp->log->write("solveCompressed: rhs for N is", rhs, hovMatrices);
  Nmat->copyFrom(*N);
  hiopVectorPar* r=rhs.new_copy(); //save the rhs to check the norm of the residual
#endif

  //
  //solve N * dyc_dyd = rhs
  //
  int ierr = solveWithRefin(*N,rhs);
  //int ierr = solve(*N,rhs);

  hiopVector& dyc_dyd= rhs;
  dyc_dyd.copyToStarting(0,           dyc);
  dyc_dyd.copyToStarting(nlp->m_eq(), dyd);

  //now solve for dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
  //first rx = -(Jc^T*dyc+Jd^T*dyd - rx)
  J.transTimesVec(1.0, rx, -1.0, dyc_dyd);
  //then dx = (H+Dx)^{-1} rx
  HessLowRank->solve(rx, dx);

#ifdef HIOP_DEEPCHECKS
  //some outputing
  nlp->log->write("KKT Low rank: solve compressed SOL", hovIteration);
  nlp->log->write("  dx: ",  dx, hovIteration); nlp->log->write(" dyc: ", dyc, hovIteration); nlp->log->write(" dyd: ", dyd, hovIteration);
  delete r;
#endif
}

int hiopKKTLinSysLowRank::solveWithRefin(hiopMatrixDense& M, hiopVectorPar& rhs)
{
  // 1. Solve dposvx (solve + equilibrating + iterative refinement + forward and backward error estimates)
  // 2. Check the residual norm
  // 3. If residual norm is not small enough, then perform iterative refinement. This is because dposvx 
  // does not always provide a small enough residual since it stops (possibly without refinement) based on
  // the forward and backward estimates

  int N=M.n();
  if(N<=0) return 0;

  hiopMatrixDense* Aref = M.new_copy();
  hiopVectorPar* rhsref = rhs.new_copy();

  char FACT='E'; 
  char UPLO='L';

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
  hiopVectorPar* x = rhs.alloc_clone(); 
  hiopVectorPar dx(N);
  hiopVectorPar resid(N); 
  int nIterRefin=0;double nrmResid;
  int info;
  const int MAX_ITER_REFIN=3;
  while(true) {
    x->copyFrom(X);
    resid.copyFrom(*rhsref);
    Aref->timesVec(1.0, resid, -1.0, *x);

    nlp->log->write("resid", resid, hovLinAlgScalars);

    nrmResid= resid.infnorm();
    nlp->log->printf(hovScalars, "hiopKKTLinSysLowRank::solveWithRefin iterrefin=%d  residual norm=%g\n", nIterRefin, nrmResid);

    if(nrmResid<1e-8) break;

    if(nIterRefin>=MAX_ITER_REFIN) {
      nlp->log->write("N", *Aref, hovMatrices);
      nlp->log->write("sol", *x, hovMatrices);
      nlp->log->write("rhs", *rhsref, hovMatrices);

      nlp->log->printf(hovWarning, "hiopKKTLinSysLowRank::solveWithRefin reduced residual to ONLY (inf-norm) %g after %d iterative refinements\n", nrmResid, nIterRefin);
      break;
      //assert(false && "too many refinements");
    }
    if(0) { //iter refin based on symmetric indefinite factorization+solve 
      

      int _V_ipiv_vec[1000]; double _V_work_vec[1000]; int lwork=1000;
      M.copyFrom(*Aref);
      DSYTRF(&UPLO, &N, M.local_buffer(), &LDA, _V_ipiv_vec, _V_work_vec, &lwork, &info);
      assert(info==0);
      DSYTRS(&UPLO, &N, &NRHS, M.local_buffer(), &LDA, _V_ipiv_vec, resid.local_data(), &LDB, &info);
      assert(info==0);
    } else { //iter refin based on symmetric positive definite factorization+solve 
      M.copyFrom(*Aref);
      //for(int i=0; i<4; i++) M.local_data()[i][i] +=1e-8;
      DPOTRF(&UPLO, &N, M.local_buffer(), &LDA, &info);
      if(info>0)
	nlp->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf (Chol fact) detected %d minor being indefinite.\n", info);
      else
	if(info<0) 
	  nlp->log->printf(hovError, "hiopKKTLinSysLowRank::factorizeMat: dpotrf returned error %d\n", info);
      
      DPOTRS(&UPLO,&N, &NRHS, M.local_buffer(), &LDA, resid.local_data(), &LDA, &info);
      if(info<0) 
	nlp->log->printf(hovError, "hiopKKTLinSysLowRank::solveWithFactors: dpotrs returned error %d\n", info);
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
    
    dx.copyFrom(resid);
    x->axpy(1., dx);
    
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

// #ifdef HIOP_DEEPCHECKS
//   hiopVectorPar sol(rhs.get_size());
//   hiopVectorPar rhss(rhs.get_size());
//   sol.copyFrom(rhs); rhss.copyFrom(*r);
//   double relErr=solveError(*Nmat, rhs, *r);
//   if(relErr>1e-5)  {
//     nlp->log->printf(hovWarning, "large rel. error (%g) in linear solver occured the Cholesky solve (hiopKKTLinSys)\n", relErr);

//     nlp->log->write("matrix N=", *Nmat, hovError);
//     nlp->log->write("rhs", rhss, hovError);
//     nlp->log->write("sol", sol, hovError);

//     assert(false && "large error (%g) in linear solve (hiopKKTLinSys), equilibrating the matrix and/or iterative refinement are needed (see dposvx/x)");
//   } else 
//     if(relErr>1e-16) 
//       nlp->log->printf(hovWarning, "considerable rel. error (%g) in linear solver occured the Cholesky solve (hiopKKTLinSys)\n", relErr);

//   nlp->log->printf(hovLinAlgScalars, "hiopKKTLinSysLowRank::solveCompressed: Cholesky solve: relative error %g\n", relErr);
//   delete r;
// #endif
  return 0;
}

int hiopKKTLinSysLowRank::solve(hiopMatrixDense& M, hiopVectorPar& rhs)
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
  nlp->log->write("Scaling S", rhs, hovSummary);

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
  //nlp->log->write("Scaling S", rhs, hovSummary);

  //M.copyFrom(AF);
  //nlp->log->write("Factoriz ", M, hovSummary);

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
errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
		      const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd)
{
  nlp->log->printf(hovLinAlgScalars, "hiopKKTLinSysLowRank::errorCompressedLinsys residuals norm:\n");

  double derr=-1., aux;
  hiopVectorPar *RX=rx.new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd
  HessLowRank->timesVec(1.0, *RX, -1.0, dx);
  //RX->axzpy(-1.0,*Dx,dx);
  Jac_c->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, "  >>>  rx=%g\n", aux);
  //if(aux>1e-8) {
  //nlp->log->write("Low rank Hessian is:", *Hess, hovLinAlgScalars); 
  //}
  delete RX; RX=NULL;

  hiopVectorPar* RC=ryc.new_copy();
  Jac_c->timesVec(1.0,*RC, -1.0,dx);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, "  >>> ryc=%g\n", aux);
  delete RC; RC=NULL;

  hiopVectorPar* RD=ryd.new_copy();
  Jac_d->timesVec(1.0,*RD, -1.0, dx);
  RD->axzpy(1.0, *Dd_inv, dyd);
  aux = RD->twonorm();
  derr=fmax(derr,aux);
  nlp->log->printf(hovLinAlgScalars, "  >>> ryd=%g\n", aux);
  delete RD; RD=NULL;

  return derr;
}

double hiopKKTLinSysLowRank::solveError(const hiopMatrixDense& M,  const hiopVectorPar& x, hiopVectorPar& rhs)
{
  double relError;
  double rhsnorm=rhs.infnorm();
  M.timesVec(1.0,rhs,-1.0,x);
  double resnorm=rhs.infnorm();
  
  relError=resnorm;// / (1+rhsnorm);
  return relError;
}
#endif
};

