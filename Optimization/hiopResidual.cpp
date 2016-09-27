#include "hiopResidual.hpp"

#include <cmath>

#include <cassert>

hiopResidual::hiopResidual(const hiopNlpDenseConstraints* nlp_)
{
  nlp = nlp_;
  rx = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  rd = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_ineq_vec());
  rxl = rx->alloc_clone();
  rxu = rx->alloc_clone();
  rdl = rd->alloc_clone();
  rdu = rd->alloc_clone();

  ryc = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_eq_vec());
  ryd = rd->alloc_clone();

  rszl = rx->alloc_clone();
  rszu = rx->alloc_clone();
  rsvl = rd->alloc_clone();
  rsvu = rd->alloc_clone();
  nrmInf_nlp_optim = nrmInf_nlp_feasib = nrmInf_nlp_complem = 1e6;
  nrmInf_bar_optim = nrmInf_bar_feasib = nrmInf_bar_complem = 1e6;
}

hiopResidual::~hiopResidual()
{
  if(rx)   delete rx;
  if(rd)   delete rd;
  if(rxl)  delete rxl;
  if(rxu)  delete rxu;
  if(rdl)  delete rdl;
  if(rdu)  delete rdu;
  if(ryc)  delete ryc;
  if(ryd)  delete ryd;
  if(rszl) delete rszl;
  if(rszu) delete rszu;
  if(rsvl) delete rsvl;
  if(rsvu) delete rsvu;
}

int hiopResidual::update(const hiopIterate& it, 
			 const double& f, const hiopVector& c, const hiopVector& d,
			 const hiopVector& grad, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
			 const double& mu)
{
  nrmInf_nlp_optim = nrmInf_nlp_feasib = nrmInf_nlp_complem = 0;
  nrmInf_bar_optim = nrmInf_bar_feasib = nrmInf_bar_complem = 0;

  long long nx_loc=rx->get_local_size();

#ifdef DEEP_CHECKING
  assert(it.zl->matchesPattern(nlp->get_ixl()));
  assert(it.zu->matchesPattern(nlp->get_ixu()));
  assert(it.sxl->matchesPattern(nlp->get_ixl()));
  assert(it.sxu->matchesPattern(nlp->get_ixu()));
#endif

  //rx = -grad_f - J_c^t*x - J_d^t*x+zl-zu
  rx->copyFrom(grad);
  jac_c.transTimesVec(1.0, *rx, 1.0, *it.yc);
  jac_d.transTimesVec(1.0, *rx, 1.0, *it.yd);
  rx->axpy(-1.0, *it.zl);
  rx->axpy( 1.0, *it.zu);
  rx->negate();
  nrmInf_nlp_optim = fmax(nrmInf_nlp_optim, rx->infnorm_local());
  //rd 
  rd->copyFrom(*it.yd);
  rd->axpy( 1.0, *it.vl);
  rd->axpy(-1.0, *it.vu);
  nrmInf_nlp_optim = fmax(nrmInf_nlp_optim, rd->infnorm_local());
  //this is also the optimality error for the log-barrier problem
  nrmInf_bar_optim = nrmInf_nlp_optim;
  //ryc
  ryc->setToConstant(0.0); 
  ryc->axpy(-1.0,c);
  nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, ryc->infnorm_local());
  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, ryd->infnorm_local());
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, rxl->infnorm_local());
  }
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, rxu->infnorm_local());
  }
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, rdl->infnorm_local());
  }
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, rdu->infnorm_local());
  }
  //set the feasibility error for the log barrier problem
  nrmInf_bar_feasib = nrmInf_nlp_feasib;
  //rszl = \mu e - sxl * zl
  if(nlp->n_low_local()>0) {
    rszl->setToZero();
    rszl->axzpy(-1.0, *it.sxl, *it.zl);
    if(nlp->n_low_local()<nx_loc)
      rszl->selectPattern(nlp->get_ixl());
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, rszl->infnorm_local());
    
    rszl->addConstant_w_patternSelect(mu,nlp->get_ixl());
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, rszl->infnorm_local());
  }
  //rszu = \mu e - sxu * zu
  if(nlp->n_upp_local()>0) {
    rszu->setToConstant(mu);
    rszl->axzpy(-1.0, *it.sxu, *it.zl);
    if(nlp->n_upp_local()<nx_loc)
      rszu->selectPattern(nlp->get_ixu());
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, rszu->infnorm_local());

    rszu->addConstant_w_patternSelect(mu,nlp->get_ixu());
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, rszu->infnorm_local());
  }
  //rsvl = \mu e - sdl * vl
  if(nlp->m_ineq_low()>0) {
    rsvl->setToConstant(mu);
    rsvl->axzpy(-1.0, *it.sdl, *it.vl);
    rsvl->selectPattern(nlp->get_idl());
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, rsvl->infnorm_local());

    rsvl->addConstant_w_patternSelect(mu,nlp->get_idl());
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, rsvl->infnorm_local());
  }
  //rsvu = \mu e - sdu * vu
  if(nlp->m_ineq_upp()>0) {
    rsvu->setToConstant(mu);
    rsvu->axzpy(-1.0, *it.sdu, *it.vu);
    rsvu->selectPattern(nlp->get_idu());
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, rsvu->infnorm_local());

    rsvu->addConstant_w_patternSelect(mu,nlp->get_idu());
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, rsvu->infnorm_local());
  }

#ifdef WITH_MPI
  //here we reduce each of the norm together for a total cost of 1 Allreduce of 3 doubles
  //otherwise, if calling infnorm() for each vector, there will be 12 Allreduce's, each of 1 double
  double aux[6]={nrmInf_nlp_optim,nrmInf_nlp_feasib,nrmInf_nlp_complem,nrmInf_bar_optim,nrmInf_bar_feasib,nrmInf_bar_complem}, aux_g[6];
  int ierr = MPI_Allreduce(aux, aux_g, 6, MPI_DOUBLE, MPI_MAX, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrmInf_nlp_optim=aux_g[0]; nrmInf_nlp_feasib=aux_g[1]; nrmInf_nlp_complem=aux_g[2];
  nrmInf_bar_optim=aux_g[3]; nrmInf_bar_feasib=aux_g[4]; nrmInf_bar_complem=aux_g[5];
#endif
  return true;
}


// void hiopResidual::
// projectPrimalsIntoBounds(double kappa1, double kappa2)
// {
//   x->projectIntoBounds(nlp->get_xl(),nlp->get_ixl(),
// 		       nlp->get_xu(),nlp->get_ixu(),
// 		       kappa1,kappa2);
//   d->projectIntoBounds(nlp->get_dl(),nlp->get_idl(),
// 		       nlp->get_du(),nlp->get_idu(),
// 		       kappa1,kappa2);
// }

// void hiopResidual::setBoundsDualsToConstant(const double& v)
// {
//   zl->setToConstant_w_patternSelect(v, nlp->get_ixl());
//   zu->setToConstant_w_patternSelect(v, nlp->get_ixu());
//   vl->setToConstant_w_patternSelect(v, nlp->get_idl());
//   vu->setToConstant_w_patternSelect(v, nlp->get_idu());
// #ifdef WITH_GPU
//   //maybe do the above arithmetically zl->setToConstant(); zl=zl.*ixl
// #endif
// }

// void hiopResidual::setEqualityDualsToConstant(const double& v)
// {
//   yc->setToConstant(v);
//   yd->setToConstant(v);
// }


// void hiopResidual::determineSlacks()
// {
//   sxl->copyFrom(*x);
//   sxl->axpy(-1., nlp->get_xl());
//   sxl->selectPattern(nlp->get_ixl());

//   sxu->copyFrom(nlp->get_xu());
//   sxu->axpy(-1., *x); 
//   sxu->selectPattern(nlp->get_ixu());

//   sdl->copyFrom(*d);
//   sdl->axpy(-1., nlp->get_dl());
//   sdl->selectPattern(nlp->get_idl());

//   sdu->copyFrom(nlp->get_du());
//   sdu->axpy(-1., *d); 
//   sdu->selectPattern(nlp->get_idu());

// #ifdef DEEP_CHECKING
//   assert(sxl->allPositive_w_patternSelect(nlp->get_ixl()));
//   assert(sxu->allPositive_w_patternSelect(nlp->get_ixu()));
//   assert(sdl->allPositive_w_patternSelect(nlp->get_idl()));
//   assert(sdu->allPositive_w_patternSelect(nlp->get_idu()));
// #endif
// }
