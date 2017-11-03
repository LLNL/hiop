#include "hiopResidual.hpp"
#include "hiopInnerProdWeight.hpp"

#include <cmath>
#include <cassert>

namespace hiop
{

hiopResidual::hiopResidual(hiopNlpDenseConstraints* nlp_)
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
  nrm_nlp_optim = nrm_nlp_feasib = nrm_nlp_complem = 1e6;
  nrm_bar_optim = nrm_bar_feasib = nrm_bar_complem = 1e6;
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

double hiopResidual::computeNlpInfeasNorm(const hiopIterate& it, 
			       const hiopVector& c, 
			       const hiopVector& d)
{
  nlp->runStats.tmSolverInternal.start();
  
  double nrmInf_infeasib;
  long long nx_loc=rx->get_local_size();
  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
  nrmInf_infeasib = ryc->infnorm();
  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  nrmInf_infeasib = fmax(nrmInf_infeasib, ryd->infnorm());
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    nrmInf_infeasib = fmax(nrmInf_infeasib, nlp->H->primalnorm(*rxl));
  }
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    nrmInf_infeasib = fmax(nrmInf_infeasib, nlp->H->primalnorm(*rxu));
  }
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rdl->infnorm());
  }
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rdu->infnorm());
  }
  //assert(false);
  nlp->runStats.tmSolverInternal.stop();
  return nrmInf_infeasib;
}

int hiopResidual::update(const hiopIterate& it, 
			 const double& f, const hiopVector& c, const hiopVector& d,
			 const hiopVector& grad, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
			 const hiopLogBarProblem& logprob)
{
  nlp->runStats.tmSolverInternal.start();

  nrm_nlp_optim = nrm_nlp_feasib = nrm_nlp_complem = 0;
  nrm_bar_optim = nrm_bar_feasib = nrm_bar_complem = 0;

  long long nx_loc=rx->get_local_size();
  const double&  mu=logprob.mu; double aux;
#ifdef DEEP_CHECKING
  assert(it.zl->matchesPattern(nlp->get_ixl()));
  assert(it.zu->matchesPattern(nlp->get_ixu()));
  assert(it.sxl->matchesPattern(nlp->get_ixl()));
  assert(it.sxu->matchesPattern(nlp->get_ixu()));
#endif

  

  // rx = -grad_f - J_c^t*x - J_d^t*x+zl-zu - linear damping term in x
  rx->copyFrom(grad);
  //nlp->H->applyInverse(*rx);

  jac_c.transTimesVec(1.0, *rx, 1.0, *it.yc);
  jac_d.transTimesVec(1.0, *rx, 1.0, *it.yd);
  //nlp->H->applyAdjoint( 1.0, *rx, jac_c, 1.0, *it.yc);
  //nlp->H->applyAdjoint( 1.0, *rx, jac_d, 1.0, *it.yd);

  //apply inverse Riesz ( fin-dim is   rx->axpy(-1.0, *it.zl); )
  //nlp->H->applyInverse(1.0, *rx, -1.0, *it.zl);
  rx->axpy(-1.0, *it.zl);
  //apply inverse Riesz (fin-dim is rx->axpy( 1.0, *it.zu); )
  //nlp->H->applyInverse(1.0, *rx,  1.0, *it.zl);
  //~nrm_nlp_optim = fmax(nrm_nlp_optim, rx->infnorm_local());
  rx->axpy( 1.0, *it.zu);

  aux = nlp->H->primalnorm(*rx);
  nrm_nlp_optim =aux;

  //nlp->log->write("*************** rx:", *rx, hovScalars);

  logprob.addNonLogBarTermsToGrad_x(1.0, *rx);
  rx->negate();
  //~nrm_bar_optim = fmax(nrm_bar_optim, rx->infnorm_local());
  nrm_bar_optim = nlp->H->primalnorm(*rx);

  nlp->log->printf(hovScalars,"resid:update: infHinv norm rx=%18.10f  logbar=%18.10f\n", aux, nrm_bar_optim);
  //~ done with rx
  //nlp->log->write("x", *it.x, hovScalars);

  //Notice: nrm_bar_optim should also incorporate the grad with respect to ineq-slack 
  //variables. We take the nrm_bar_optim = max( \|rx\|_{U^*}, \|rd\|_\infty ). This is
  //not in the user manual or any hiop papers. 

  // rd 
  rd->copyFrom(*it.yd);
  rd->axpy( 1.0, *it.vl);
  rd->axpy(-1.0, *it.vu);
  //~! nrm_nlp_optim = fmax(nrm_nlp_optim, rd->infnorm_local());
  aux = rd->infnorm();
  nrm_nlp_optim = fmax(nrm_nlp_optim, aux); 
  nlp->log->printf(hovScalars,"resid:update: inf_Hinv norm rd=%g ; inf component of it is %g\n", rd->infnorm_local(), aux);
  logprob.addNonLogBarTermsToGrad_d(-1.0,*rd);
  //~! nrm_bar_optim = fmax(nrm_bar_optim, rd->infnorm_local());
  nrm_bar_optim = fmax(nrm_bar_optim, rd->infnorm()); 

  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
  //~! nrm_nlp_feasib = fmax(nrm_nlp_feasib, ryc->infnorm_local());
  aux = ryc->infnorm();
  nrm_nlp_feasib = aux;
  nlp->log->printf(hovScalars,"resid:update: inf norm ryc=%g\n", aux);

  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  aux = ryd->infnorm();
  nrm_nlp_feasib = fmax(nrm_nlp_feasib, aux);
  nlp->log->printf(hovScalars,"resid:update: inf norm ryd=%g\n", aux);

  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    //~! nrm_nlp_feasib = fmax(nrm_nlp_feasib, rxl->infnorm_local());
    aux = nlp->H->primalnorm(*rxl);
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, aux);
    nlp->log->printf(hovScalars,"resid:update: H norm rxl=%g\n", aux);
  }

  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    //~! nrm_nlp_feasib = fmax(nrm_nlp_feasib, rxu->infnorm_local());
    aux = nlp->H->primalnorm(*rxu);
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, aux);
    nlp->log->printf(hovScalars,"resid:update: H norm rxu=%g\n", aux);
  }  

  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    //~! nrm_nlp_feasib = fmax(nrm_nlp_feasib, rdl->infnorm_local());
    aux = rdl->infnorm();
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, aux);
    nlp->log->printf(hovScalars,"resid:update: inf norm rdl=%g\n", aux);
  }

  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    //~! nrm_nlp_feasib = fmax(nrm_nlp_feasib, rdu->infnorm_local());
    aux = rdu->infnorm();
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, aux);
    nlp->log->printf(hovScalars,"resid:update: inf norm rdl=%g\n", aux);
  }

  //set the feasibility error for the log barrier problem
  nrm_bar_feasib = nrm_nlp_feasib;

  //rszl = \mu e - sxl * zl
  if(nlp->n_low_local()>0) {
    rszl->setToZero();
    rszl->axzpy(-1.0, *it.sxl, *it.zl);
    if(nlp->n_low_local()<nx_loc)
      rszl->selectPattern(nlp->get_ixl());
    //~! nrm_nlp_complem = fmax(nrm_nlp_complem, rszl->infnorm_local());

    //nlp->log->write("rszl", *rszl, hovScalars);

    aux = nlp->H->primalnorm(*rszl);
    nrm_nlp_complem = fmax(nrm_nlp_complem, aux);
    
    rszl->addConstant_w_patternSelect(mu,nlp->get_ixl());
    //~! nrm_bar_complem = fmax(nrm_bar_complem, rszl->infnorm_local());
    nrm_bar_complem = fmax(nrm_bar_complem, nlp->H->primalnorm(*rszl));
    nlp->log->printf(hovScalars,"resid:update: H norm rszl=%g\n", aux);
  }

  //rszu = \mu e - sxu * zu
  if(nlp->n_upp_local()>0) {
    rszu->setToZero();
    rszu->axzpy(-1.0, *it.sxu, *it.zu);
    if(nlp->n_upp_local()<nx_loc)
      rszu->selectPattern(nlp->get_ixu());
    //~! nrm_nlp_complem = fmax(nrm_nlp_complem, rszu->infnorm_local());
    aux = nlp->H->primalnorm(*rszu);
    nrm_nlp_complem = fmax(nrm_nlp_complem, aux);

    rszu->addConstant_w_patternSelect(mu,nlp->get_ixu());
    //~! nrm_bar_complem = fmax(nrm_bar_complem, rszu->infnorm_local());
    nrm_bar_complem = fmax(nrm_bar_complem, nlp->H->primalnorm(*rszu));
    nlp->log->printf(hovScalars,"resid:update: H norm rszu=%g\n", aux);
  }

  //rsvl = \mu e - sdl * vl
  if(nlp->m_ineq_low()>0) {
    rsvl->setToZero();
    rsvl->axzpy(-1.0, *it.sdl, *it.vl);
    if(nlp->m_ineq_low()<nlp->m_ineq()) rsvl->selectPattern(nlp->get_idl());
    //~! nrm_nlp_complem = fmax(nrm_nlp_complem, rsvl->infnorm_local());
    aux = nlp->H->primalnorm(*rsvl);
    nrm_nlp_complem = fmax(nrm_nlp_complem, aux);
    //aux = rsvl->infnorm();
    //add mu
    rsvl->addConstant_w_patternSelect(mu,nlp->get_idl());
    //~! nrm_bar_complem = fmax(nrm_bar_complem, rsvl->infnorm_local());
    nrm_bar_complem = fmax(nrm_nlp_complem, rsvl->infnorm());
    nlp->log->printf(hovScalars,"resid:update: H norm rsvl=%g\n", aux);
  }

  //rsvu = \mu e - sdu * vu
  if(nlp->m_ineq_upp()>0) {
    rsvu->setToZero();
    rsvu->axzpy(-1.0, *it.sdu, *it.vu);
    if(nlp->m_ineq_upp()<nlp->m_ineq()) rsvu->selectPattern(nlp->get_idu());
    //~! nrm_nlp_complem = fmax(nrm_nlp_complem, rsvu->infnorm_local());
    aux = nlp->H->primalnorm(*rsvu);
    nrm_nlp_complem = fmax(nrm_nlp_complem, aux);

    //add mu
    rsvu->addConstant_w_patternSelect(mu,nlp->get_idu());
    //~! nrm_bar_complem = fmax(nrm_bar_complem, rsvu->infnorm_local());
    nrm_bar_complem = fmax(nrm_bar_complem, rsvu->infnorm());
    nlp->log->printf(hovScalars,"resid:update: H norm rsvu=%g\n", aux);
  }

  nlp->runStats.tmSolverInternal.stop();
  return true;
}

void hiopResidual::print(FILE* f, const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const
{
  if(NULL==msg) fprintf(f, "hiopResidual print\n");
  else fprintf(f, "%s\n", msg);

  rx->print(  f, "    rx:", max_elems, rank); 
  rd->print(  f, "    rd:", max_elems, rank);   
  ryc->print( f, "   ryc:", max_elems, rank); 
  ryd->print( f, "   ryd:", max_elems, rank); 
  rszl->print(f, "  rszl:", max_elems, rank); 
  rszu->print(f, "  rszu:", max_elems, rank); 
  rsvl->print(f, "  rsvl:", max_elems, rank); 
  rsvu->print(f, "  rsvu:", max_elems, rank); 
  rxl->print( f, "   rxl:", max_elems, rank);  
  rxu->print( f, "   rxu:", max_elems, rank); 
  rdl->print( f, "   rdl:", max_elems, rank); 
  rdu->print( f, "   rdu:", max_elems, rank); 
  printf(" errors (optim/feasib/complem) nlp    : %26.16e %25.16e %25.16e\n", 
	 nrm_nlp_optim, nrm_nlp_feasib, nrm_nlp_complem);
  printf(" errors (optim/feasib/complem) barrier: %25.16e %25.16e %25.16e\n", 
	 nrm_bar_optim, nrm_bar_feasib, nrm_bar_complem);
}


/***** Finite-dimensional implementation */
double hiopResidualFinDimImpl::computeNlpInfeasNorm(const hiopIterate& it, 
			       const hiopVector& c, 
			       const hiopVector& d)
{
  nlp->runStats.tmSolverInternal.start();
  
  double nrmInf_infeasib;
  long long nx_loc=rx->get_local_size();
  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
  nrmInf_infeasib = ryc->infnorm_local();
  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  nrmInf_infeasib = fmax(nrmInf_infeasib, ryd->infnorm_local());
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rxl->infnorm_local());
  }
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rxu->infnorm_local());
  }
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rdl->infnorm_local());
  }
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    nrmInf_infeasib = fmax(nrmInf_infeasib, rdu->infnorm_local());
  }

#ifdef WITH_MPI
  //here we reduce each of the norm together for a total cost of 1 Allreduce of 3 doubles
  //otherwise, if calling infnorm() for each vector, there will be 12 Allreduce's, each of 1 double
  double aux;
  int ierr = MPI_Allreduce(&nrmInf_infeasib, &aux, 1, MPI_DOUBLE, MPI_MAX, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrmInf_infeasib = aux;
#endif
  nlp->runStats.tmSolverInternal.stop();
  return nrmInf_infeasib;
}

int hiopResidualFinDimImpl::update(const hiopIterate& it, 
			 const double& f, const hiopVector& c, const hiopVector& d,
			 const hiopVector& grad, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
			 const hiopLogBarProblem& logprob)
{
  nlp->runStats.tmSolverInternal.start();

  nrm_nlp_optim = nrm_nlp_feasib = nrm_nlp_complem = 0;
  nrm_bar_optim = nrm_bar_feasib = nrm_bar_complem = 0;

  long long nx_loc=rx->get_local_size();
  const double&  mu=logprob.mu;
#ifdef DEEP_CHECKING
  assert(it.zl->matchesPattern(nlp->get_ixl()));
  assert(it.zu->matchesPattern(nlp->get_ixu()));
  assert(it.sxl->matchesPattern(nlp->get_ixl()));
  assert(it.sxu->matchesPattern(nlp->get_ixu()));
#endif
  // rx = -grad_f - J_c^t*x - J_d^t*x+zl-zu - linear damping term in x
  rx->copyFrom(grad);
  jac_c.transTimesVec(1.0, *rx, 1.0, *it.yc);
  jac_d.transTimesVec(1.0, *rx, 1.0, *it.yd);
  rx->axpy(-1.0, *it.zl);
  rx->axpy( 1.0, *it.zu);
  nrm_nlp_optim = fmax(nrm_nlp_optim, rx->infnorm_local());
  //nlp->log->printf(hovScalars,"resid:update: inf norm rx=%g\n", rx->infnorm_local());
  logprob.addNonLogBarTermsToGrad_x(1.0, *rx);
  rx->negate();
  nrm_bar_optim = fmax(nrm_bar_optim, rx->infnorm_local());

  //~ done with rx
  // rd 
  rd->copyFrom(*it.yd);
  rd->axpy( 1.0, *it.vl);
  rd->axpy(-1.0, *it.vu);
  nrm_nlp_optim = fmax(nrm_nlp_optim, rd->infnorm_local());
  nlp->log->printf(hovScalars,"resid:update: inf norm rd=%g\n", rd->infnorm_local());
  logprob.addNonLogBarTermsToGrad_d(-1.0,*rd);
  nrm_bar_optim = fmax(nrm_bar_optim, rd->infnorm_local());

  

  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
  nrm_nlp_feasib = fmax(nrm_nlp_feasib, ryc->infnorm_local());
  nlp->log->printf(hovScalars,"resid:update: inf norm ryc=%g\n", ryc->infnorm_local());

  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  nrm_nlp_feasib = fmax(nrm_nlp_feasib, ryd->infnorm_local());
  nlp->log->printf(hovScalars,"resid:update: inf norm ryd=%g\n", ryd->infnorm_local());
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, rxl->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rxl=%g\n", rxl->infnorm_local());
  }
  //printf("  %10.4e (xl)", nrm_nlp_feasib);
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, rxu->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rxu=%g\n", rxu->infnorm_local());
  }  
  //printf("  %10.4e (xu)", nrm_nlp_feasib);
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, rdl->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rdl=%g\n", rdl->infnorm_local());
  }
  //printf("  %10.4e (dl)", nrm_nlp_feasib);
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    nrm_nlp_feasib = fmax(nrm_nlp_feasib, rdu->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rdl=%g\n", rdu->infnorm_local());
  }
  //printf("  %10.4e (du)\n", nrm_nlp_feasib);
  //set the feasibility error for the log barrier problem
  nrm_bar_feasib = nrm_nlp_feasib;

  //rszl = \mu e - sxl * zl
  if(nlp->n_low_local()>0) {
    rszl->setToZero();
    rszl->axzpy(-1.0, *it.sxl, *it.zl);
    if(nlp->n_low_local()<nx_loc)
      rszl->selectPattern(nlp->get_ixl());
    nrm_nlp_complem = fmax(nrm_nlp_complem, rszl->infnorm_local());
    
    rszl->addConstant_w_patternSelect(mu,nlp->get_ixl());
    nrm_bar_complem = fmax(nrm_bar_complem, rszl->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rszl=%g\n", rszl->infnorm_local());
  }
  //rszu = \mu e - sxu * zu
  if(nlp->n_upp_local()>0) {
    rszu->setToZero();
    rszu->axzpy(-1.0, *it.sxu, *it.zu);
    if(nlp->n_upp_local()<nx_loc)
      rszu->selectPattern(nlp->get_ixu());
    nrm_nlp_complem = fmax(nrm_nlp_complem, rszu->infnorm_local());

    rszu->addConstant_w_patternSelect(mu,nlp->get_ixu());
    nrm_bar_complem = fmax(nrm_bar_complem, rszu->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rszu=%g\n", rszu->infnorm_local());
  }
  //rsvl = \mu e - sdl * vl
  if(nlp->m_ineq_low()>0) {
    rsvl->setToZero();
    rsvl->axzpy(-1.0, *it.sdl, *it.vl);
    if(nlp->m_ineq_low()<nlp->m_ineq()) rsvl->selectPattern(nlp->get_idl());
    nrm_nlp_complem = fmax(nrm_nlp_complem, rsvl->infnorm_local());

    //add mu
    rsvl->addConstant_w_patternSelect(mu,nlp->get_idl());
    nrm_bar_complem = fmax(nrm_bar_complem, rsvl->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rsvl=%g\n", rsvl->infnorm_local());
  }
  //rsvu = \mu e - sdu * vu
  if(nlp->m_ineq_upp()>0) {
    rsvu->setToZero();
    rsvu->axzpy(-1.0, *it.sdu, *it.vu);
    if(nlp->m_ineq_upp()<nlp->m_ineq()) rsvu->selectPattern(nlp->get_idu());
    nrm_nlp_complem = fmax(nrm_nlp_complem, rsvu->infnorm_local());

    //add mu
    rsvu->addConstant_w_patternSelect(mu,nlp->get_idu());
    nrm_bar_complem = fmax(nrm_bar_complem, rsvu->infnorm_local());
    nlp->log->printf(hovScalars,"resid:update: inf norm rsvu=%g\n", rsvu->infnorm_local());
  }

#ifdef WITH_MPI
  //here we reduce each of the norm together for a total cost of 1 Allreduce of 3 doubles
  //otherwise, if calling infnorm() for each vector, there will be 12 Allreduce's, each of 1 double
  double aux[6]={nrm_nlp_optim,nrm_nlp_feasib,nrm_nlp_complem,nrm_bar_optim,nrm_bar_feasib,nrm_bar_complem}, aux_g[6];
  int ierr = MPI_Allreduce(aux, aux_g, 6, MPI_DOUBLE, MPI_MAX, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrm_nlp_optim=aux_g[0]; nrm_nlp_feasib=aux_g[1]; nrm_nlp_complem=aux_g[2];
  nrm_bar_optim=aux_g[3]; nrm_bar_feasib=aux_g[4]; nrm_bar_complem=aux_g[5];
#endif
  nlp->runStats.tmSolverInternal.stop();
  return true;
}


}; //end of namespace
