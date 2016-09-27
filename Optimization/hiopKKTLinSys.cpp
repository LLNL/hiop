#include "hiopKKTLinSys.hpp"
#include "blasdefs.hpp"

#include <cmath>

hiopKKTLinSysLowRank::hiopKKTLinSysLowRank(const hiopNlpFormulation* nlp_)
{
  iter=NULL; grad_f=NULL; Jac_c=Jac_d=NULL; Hess=NULL;
  nlp = dynamic_cast<const hiopNlpDenseConstraints*>(nlp_);
  rx_tilde  = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  Dx = rx_tilde->alloc_clone();
  ryd_tilde = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_ineq_vec());
  Dd_inv = ryd_tilde->alloc_clone();
  _kxn_mat = nlp->alloc_multivector_primal(nlp->m()); //!opt
  N = new hiopMatrixDense(nlp->m(),nlp->m());
#ifdef DEEP_CHECKING
  Nmat=N->alloc_clone();
#endif
  _k_vec1 = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_vec());
}

hiopKKTLinSysLowRank::~hiopKKTLinSysLowRank()
{
  if(rx_tilde)  delete rx_tilde;
  if(ryd_tilde) delete ryd_tilde;
  if(N)         delete N;
#ifdef DEEP_CHECKING
  if(Nmat)      delete Nmat;
#endif
  if(Dx)        delete Dx;
  if(Dd_inv)    delete Dd_inv;
  if(_kxn_mat)  delete _kxn_mat;
  if(_k_vec1)   delete _k_vec1;
}

bool hiopKKTLinSysLowRank::
update(const hiopIterate* iter_, 
       const hiopVector* grad_f_, 
       const hiopMatrixDense* Jac_c_, const hiopMatrixDense* Jac_d_, 
       hiopHessianLowRank* Hess_)
{
  iter=iter_;
  grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_);
  Jac_c = Jac_c_; Jac_d = Jac_d_;
  Hess = dynamic_cast<hiopHessianInvLowRank*>(Hess_);

  //compute the diagonals
  //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx->setToZero();
  Dx->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp->get_ixl());
  Dx->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp->get_ixu());
  Dx->print("Dx in KKT");

  Hess->updateDiagonal(*Dx);

  //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_inv->setToZero();
  Dd_inv->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
  Dd_inv->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef DEEP_CHECKING
  assert(true==Dd_inv->allPositive());
#endif 
  Dd_inv->invert();
  return true;
}

bool hiopKKTLinSysLowRank::computeDirections(const hiopResidual* resid, 
					     hiopIterate* dir)
{
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

  Dd_inv->print("Dinv ");

  //now the final ryd_tilde += Dd^{-1}*ryd2
  ryd_tilde->axzpy(1.0, ryd2, *Dd_inv);

#ifdef DEEP_CHECKING
  hiopVectorPar* rx_tilde_save=rx_tilde->new_copy();
  hiopVectorPar* ryc_save=r.ryc->new_copy();
  hiopVectorPar* ryd_tilde_save=ryd_tilde->new_copy();
#endif

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function) 
   */
  solveCompressed(*rx_tilde,*r.ryc,*ryd_tilde, *dir->x, *dir->yc, *dir->yd);
  //recover dir->d = (D)^{-1}*(dir->yd + ryd2)
  dir->d->copyFrom(ryd2);
  dir->d->axpy(1.0,*dir->yd);
  dir->d->componentMult(*Dd_inv);
  //dir->d->axzpy(1.0, ryd2, *Dd_inv);

  //dir->d->print();

#ifdef DEEP_CHECKING
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
  }
  //dir->sxl->print();
  //dir->zl->print();
  //dsxu = rxu - dx and dzu = [Sxu]^{-1} ( - Zu*dsxu + rszu)
  if(nlp->n_upp_local()) { 
    dir->sxu->copyFrom(*r.rxu); dir->sxu->axpy(-1.0,*dir->x); dir->sxu->selectPattern(nlp->get_ixu()); 

    dir->zu->copyFrom(*r.rszu); dir->zu->axzpy(-1.0,*iter->zu,*dir->sxu); dir->zu->selectPattern(nlp->get_ixu());
    dir->zu->componentDiv_p_selectPattern(*iter->sxu, nlp->get_ixu());
  }
  //dir->sxu->print();
  //dir->zu->print();
  //dsdl = rdl + dd and dvl = [Sdl]^{-1} ( - Vl*dsdl + rsvl)
  if(nlp->m_ineq_low()) {
    dir->sdl->copyFrom(*r.rdl); dir->sdl->axpy( 1.0,*dir->d); dir->sdl->selectPattern(nlp->get_idl());

    dir->vl->copyFrom(*r.rsvl); dir->vl->axzpy(-1.0,*iter->vl,*dir->sdl); dir->vl->selectPattern(nlp->get_idl());
    dir->vl->componentDiv_p_selectPattern(*iter->sdl, nlp->get_idl());
  }
  //dir->sdl->print();
  // dir->vl->print();
  //dsdu = rdu - dd and dvu = [Sdu]^{-1} ( - Vu*dsdu + rsvu )
  if(nlp->m_ineq_upp()>0) {
    dir->sdu->copyFrom(*r.rdu); dir->sdu->axpy(-1.0,*dir->d); dir->sdu->selectPattern(nlp->get_idu());
    
    dir->vu->copyFrom(*r.rsvu); dir->vu->axzpy(-1.0,*iter->vu,*dir->sdu); dir->vu->selectPattern(nlp->get_idu());
    dir->vu->componentDiv_p_selectPattern(*iter->sdu, nlp->get_idu());
  }
  //dir->sdu->print();
  //dir->vu->print();
#ifdef DEEP_CHECKING
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

  return true;
}

#ifdef DEEP_CHECKING
double hiopKKTLinSysLowRank::errorKKT(const hiopResidual* resid, const hiopIterate* sol)
{
  double derr=1e20,aux;
  hiopVectorPar *RX=resid->rx->new_copy();
  //RX=rx-H*dx-J'c*dyc-J'*dyd -dzl-dzu = rx
  Hess->timesVec(1.0, *RX, -1.0, *sol->x);
  Jac_c->transTimesVec(1.0, *RX, -1.0, *sol->yc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, *sol->yd);
  sol->zl->print("zl");
  sol->zu->print("zu");
  RX->axpy( 1.0,*sol->zl);
  RX->axpy(-1.0,*sol->zu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rx=%g\n", aux);

  hiopVectorPar* RYC=resid->ryc->new_copy();
  Jac_c->timesVec(1.0, *RYC, -1.0, *sol->x);
  aux=RYC->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: ryc=%g\n", aux);
  delete RYC;

  //RYD=ryd-Jd*dx+dd
  hiopVectorPar* RYD=resid->ryd->new_copy();
  Jac_d->timesVec(1.0, *RYD, -1.0, *sol->x);
  RYD->axpy(1.0,*sol->d);
  aux=RYD->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: ryd=%g\n", aux);
  delete RYD; 

  //RXL=rxl+x-sxl
  RX->copyFrom(*resid->rxl);
  RX->axpy( 1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxl);
  RX->selectPattern(nlp->get_ixl());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rxl=%g\n", aux);
  //RXU=rxu-x-sxu
  RX->copyFrom(*resid->rxu);
  RX->axpy(-1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxu);
  RX->selectPattern(nlp->get_ixu());
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rxu=%g\n", aux);
 

  //RDL=rdl+d-sdl
  hiopVectorPar* RD=resid->rdl->new_copy();
  RD->axpy( 1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdl);
  RD->selectPattern(nlp->get_idl());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rdl=%g\n", aux);
  //RDU=rdu-d-sdu
  RD->copyFrom(*resid->rdu);
  RD->axpy(-1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdu);
  RD->selectPattern(nlp->get_idu());
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rdl=%g\n", aux);

  
  //complementarity residuals checks: rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszl);
  RX->axzpy(-1.0,*iter->sxl,*sol->zl);
  RX->axzpy(-1.0,*iter->zl, *sol->sxl);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rszl=%g\n", aux);
  //rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszu);
  RX->axzpy(-1.0,*iter->sxu,*sol->zu);
  RX->axzpy(-1.0,*iter->zu, *sol->sxu);
  aux=RX->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rszu=%g\n", aux);
  delete RX; RX=NULL;
 
  //complementarity residuals checks: rsvl - Sdl dvl - Vl dsdl
  RD->copyFrom(*resid->rsvl);
  RD->axzpy(-1.0,*iter->sdl,*sol->vl);
  RD->axzpy(-1.0,*iter->vl, *sol->sdl);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rsvl=%g\n", aux);
  //complementarity residuals checks: rsvu - Sdu dvu - Vu dsdu
  RD->copyFrom(*resid->rsvu);
  RD->axzpy(-1.0,*iter->sdu,*sol->vu);
  RD->axzpy(-1.0,*iter->vu, *sol->sdu);
  aux=RD->twonorm();
  derr=fmax(aux,derr);
  printf("error check: KKT Linsys: rsvu=%g\n", aux);

  delete RD; RD=NULL;
  return derr;
}
double hiopKKTLinSysLowRank::
errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
		      const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd)
{
  double derr=1e20, aux;
  hiopVectorPar *RX=rx.new_copy();
  //RX=rx-H*dx-Dx*dx-J'c*dyc-J'*dyd
  Hess->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0,*Dx,dx);
  Jac_c->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d->transTimesVec(1.0, *RX, -1.0, dyd);
  aux=RX->twonorm();
  derr=fmax(derr,aux);
  printf("error check: compressedLinsys: rx=%g\n", aux);
  delete RX; RX=NULL;

  hiopVectorPar* RC=ryc.new_copy();
  Jac_c->timesVec(1.0,*RC, -1.0,dx);
  aux = RC->twonorm();
  derr=fmax(derr,aux);
  printf("error check: compressedLinsys: ryc=%g\n", aux);
  delete RC; RC=NULL;

  hiopVectorPar* RD=ryd.new_copy();
  Jac_d->timesVec(1.0,*RD, -1.0, dx);
  RD->axzpy(1.0, *Dd_inv, dyd);
  aux = RD->twonorm();
  derr=fmax(derr,aux);
  printf("error check: compressedLinsys: ryd=%g\n", aux);
  delete RD; RD=NULL;

  return derr;
}
#endif
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
   */
void hiopKKTLinSysLowRank::
solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
		hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
{
  hiopMatrixDense& J = *_kxn_mat;
  J.copyRowsFrom(*Jac_c, nlp->m_eq(), 0); //!opt
  J.copyRowsFrom(*Jac_d, nlp->m_ineq(), nlp->m_eq());//!opt

  Hess->symmetricTimesMat(0.0, *N, 1.0, J);

  N->addSubDiagonal(nlp->m_eq(), *Dd_inv);
#ifdef DEEP_CHECKING
  N->assertSymmetry();
  Nmat->copyFrom(*N);
#endif
  int ierr=factorizeMat(*N); assert(ierr==0);

  //compute the rhs of the lin sys involving N 
  //  first compute (H+Dx)^{-1} rx_tilde and store it temporarily in dx
  Hess->apply(0.0, dx, 1.0, rx);
  // then rhs =   [ Jc(H+Dx)^{-1}*rx - ryc ]
  //              [ Jd(H+dx)^{-1}*rx - ryd ]
  hiopVectorPar& rhs=*_k_vec1;
  rhs.copyFromStarting(ryc,0);
  rhs.copyFromStarting(ryd,nlp->m_eq());
  J.timesVec(-1.0, rhs, 1.0, dx);

#ifdef DEEP_CHECKING
  hiopVectorPar* r=rhs.new_copy(); //save the rhs to check the residual
#endif

  //solve N * dyc_dyd = rhs
  ierr=solveWithFactors(*N,rhs); assert(ierr==0);

#ifdef DEEP_CHECKING
  double relErr=solveError(*Nmat, rhs, *r);
  if(relErr>1e-4) assert(false && "large error (%g) in linear solve (hiopKKTLinSys), equilibrating the matrix and/or iterative refinement are needed (see dposvx/x)");
  else if(relErr>1e-7) printf("log:warn: considerable error (%g) in linear solver occured (hiopKKTLinSys)", relErr);
  delete r;
#endif
  hiopVector& dyc_dyd= rhs;
  dyc_dyd.copyToStarting(dyc,0);
  dyc_dyd.copyToStarting(dyd,nlp->m_eq());

  //now solve for dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
  //first rx = -(Jc^T*dyc+Jd^T*dyd - rx)
  J.transTimesVec(1.0, rx, -1.0, dyc_dyd);
  //then dx = (H+Dx)^{-1} rx
  Hess->apply(0.0, dx, 1.0, rx);
}
int hiopKKTLinSysLowRank::factorizeMat(hiopMatrixDense& M)
{
#ifdef DEEP_CHECKING
  assert(M.m()==M.n());
#endif
  if(M.m()==0) return 0;
  char uplo='L'; int N=M.n(), lda=N, info;
  dpotrf_(&uplo, &N, M.local_buffer(), &lda, &info);
  if(info>0)
    printf("log:warn: dpotrf (Chol fact) detected %d minor being indefinite.\n", info);
  else
    if(info<0) printf("log:err: dpotrf returned error %d\n", info);
  return info;
}

int hiopKKTLinSysLowRank::solveWithFactors(hiopMatrixDense& M, hiopVectorPar& r)
{
#ifdef DEEP_CHECKING
  assert(M.m()==M.n());
#endif
  if(M.m()==0) return 0;
  char uplo='L'; //we have upper triangular in C++, but this is lower in fortran
  int N=M.n(), lda=N, nrhs=1, info;
  dpotrs_(&uplo,&N, &nrhs, M.local_buffer(), &lda, r.local_data(), &lda, &info);
  if(info<0) printf("log:err: dpotrs returned error %d\n", info);
#ifdef DEEP_CHECKING
  assert(info<=0);
#endif
  return info;
}

#ifdef DEEP_CHECKING
double hiopKKTLinSysLowRank::solveError(const hiopMatrixDense& M,  const hiopVectorPar& x, hiopVectorPar& rhs)
{
  double relError;
  M.timesVec(1.0,rhs,-1.0,x);
  double xnorm=x.twonorm();
  double rnorm=rhs.twonorm();
  relError=rnorm/(1+xnorm);
  return relError;
}
#endif
