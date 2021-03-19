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

#include "hiopResidual.hpp"

#include <cmath>
#include <cassert>

namespace hiop
{

hiopResidual::hiopResidual(hiopNlpFormulation* nlp_)
{
  nlp = nlp_;
  rx = nlp->alloc_primal_vec();
  rd = nlp->alloc_dual_ineq_vec();
  rxl = rx->alloc_clone();
  rxu = rx->alloc_clone();
  rdl = rd->alloc_clone();
  rdu = rd->alloc_clone();

  ryc = nlp->alloc_dual_eq_vec();
  ryd = rd->alloc_clone();

  rszl = rx->alloc_clone();
  rszl->setToZero();
  rszu = rszl->new_copy();

  rsvl = rd->alloc_clone();
  rsvl->setToZero();
  rsvu = rsvl->new_copy();

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

double hiopResidual::compute_nlp_norms(const hiopIterate& it, 
			       const hiopVector& c, 
			       const hiopVector& d)
{
  nlp->runStats.tmSolverInternal.start();
  
//  double nrmInf_infeasib;
  double nrmOne_infeasib = 0.;
  long long nx_loc=rx->get_local_size();
  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
//  nrmInf_infeasib = ryc->infnorm_local();
  nrmOne_infeasib += ryc->onenorm_local();
  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
//  nrmInf_infeasib = fmax(nrmInf_infeasib, ryd->infnorm_local());
  nrmOne_infeasib += ryd->onenorm_local();
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
//    nrmInf_infeasib = fmax(nrmInf_infeasib, rxl->infnorm_local());
  }
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu()); rxu->axpy(-1.0,*it.x); rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
//    nrmInf_infeasib = fmax(nrmInf_infeasib, rxu->infnorm_local());
  }
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
//    nrmInf_infeasib = fmax(nrmInf_infeasib, rdl->infnorm_local());
  }
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du()); rdu->axpy(-1.0,*it.sdu); rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
//    nrmInf_infeasib = fmax(nrmInf_infeasib, rdu->infnorm_local());
  }

#ifdef HIOP_USE_MPI
  //here we reduce each of the norm together for a total cost of 1 Allreduce of 3 doubles
  //otherwise, if calling infnorm() for each vector, there will be 12 Allreduce's, each of 1 double
//  double aux;
//  int ierr = MPI_Allreduce(&nrmInf_infeasib, &aux, 1, MPI_DOUBLE, MPI_MAX, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
//  nrmInf_infeasib = aux;

  double sum_one_norm = 0.0;
  int ierr = MPI_Allreduce(nrmOne_infeasib, sum_one_norm, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrmOne_infeasib = sum_one_norm;
#endif
  nlp->runStats.tmSolverInternal.stop();
  return nrmOne_infeasib;
}

int hiopResidual::update(const hiopIterate& it, 
			 const double& f, const hiopVector& c, const hiopVector& d,
			 const hiopVector& grad, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
			 const hiopLogBarProblem& logprob)
{
  nlp->runStats.tmSolverInternal.start();

  nrmInf_nlp_optim = nrmInf_nlp_feasib = nrmInf_nlp_complem = 0;
  nrmInf_bar_optim = nrmInf_bar_feasib = nrmInf_bar_complem = 0;
  nrmOne_theta = 0.;

  long long nx_loc=rx->get_local_size();
  const double&  mu=logprob.mu;
  double buf;
#ifdef HIOP_DEEPCHECKS
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
  buf = rx->infnorm_local();
  nrmInf_nlp_optim = fmax(nrmInf_nlp_optim, buf);
  nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rx=%22.17e\n", buf);
  logprob.addNonLogBarTermsToGrad_x(1.0, *rx);
  rx->negate();
  nrmInf_bar_optim = fmax(nrmInf_bar_optim, rx->infnorm_local());
  //~ done with rx
  // rd 
  rd->copyFrom(*it.yd);
  rd->axpy( 1.0, *it.vl);
  rd->axpy(-1.0, *it.vu);
  buf = rd->infnorm_local();
  nrmInf_nlp_optim = fmax(nrmInf_nlp_optim, buf);
  nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rd=%22.17e\n", buf);
  logprob.addNonLogBarTermsToGrad_d(-1.0,*rd);
  nrmInf_bar_optim = fmax(nrmInf_bar_optim, rd->infnorm_local());
  //ryc
  ryc->copyFrom(nlp->get_crhs());
  ryc->axpy(-1.0,c);
  buf = ryc->infnorm_local();
  nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
  nrmOne_theta += ryc->onenorm_local();
  nlp->log->printf(hovScalars,"NLP resid [update]: inf norm ryc=%22.17e\n", buf);

  //ryd
  ryd->copyFrom(*it.d);
  ryd->axpy(-1.0, d);
  buf = ryd->infnorm_local();
  nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
  nrmOne_theta += ryd->onenorm_local();
  nlp->log->printf(hovScalars,"NLP resid [update]: inf norm ryd=%22.17e\n", buf);
  
  //rxl=x-sxl-xl
  if(nlp->n_low_local()>0) {
    rxl->copyFrom(*it.x);
    rxl->axpy(-1.0,*it.sxl);
    rxl->axpy(-1.0,nlp->get_xl());
    //zero out entries in the resid that don't correspond to a finite low bound 
    if(nlp->n_low_local()<nx_loc)
      rxl->selectPattern(nlp->get_ixl());
    buf = rxl->infnorm_local();
//    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rxl=%22.17e\n", buf);
  }
  //printf("  %10.4e (xl)", nrmInf_nlp_feasib);
  //rxu=-x-sxu+xu
  if(nlp->n_upp_local()>0) {
    rxu->copyFrom(nlp->get_xu());
    rxu->axpy(-1.0,*it.x);
    rxu->axpy(-1.0,*it.sxu);
    if(nlp->n_upp_local()<nx_loc)
      rxu->selectPattern(nlp->get_ixu());
    buf = rxu->infnorm_local();
//    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rxu=%22.17e\n", buf);
  }  
  //printf("  %10.4e (xu)", nrmInf_nlp_feasib);
  //rdl=d-sdl-dl
  if(nlp->m_ineq_low()>0) {
    rdl->copyFrom(*it.d); rdl->axpy(-1.0,*it.sdl); rdl->axpy(-1.0,nlp->get_dl());
    rdl->selectPattern(nlp->get_idl());
    buf = rdl->infnorm_local();
//    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rdl=%22.17e\n", buf);
  }
  //printf("  %10.4e (dl)", nrmInf_nlp_feasib);
  //rdu=-d-sdu+du
  if(nlp->m_ineq_upp()>0) {
    rdu->copyFrom(nlp->get_du());
    rdu->axpy(-1.0,*it.sdu);
    rdu->axpy(-1.0,*it.d);
    rdu->selectPattern(nlp->get_idu());
    buf = rdu->infnorm_local();
//    nrmInf_nlp_feasib = fmax(nrmInf_nlp_feasib, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rdl=%22.17e\n", buf);
  }
  //printf("  %10.4e (du)\n", nrmInf_nlp_feasib);
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
    buf = rszl->infnorm_local();
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rszl=%22.17e\n", buf);
  }
  //rszu = \mu e - sxu * zu
  if(nlp->n_upp_local()>0) {
    rszu->setToZero();
    rszu->axzpy(-1.0, *it.sxu, *it.zu);
    if(nlp->n_upp_local()<nx_loc)
      rszu->selectPattern(nlp->get_ixu());

    buf = rszu->infnorm_local();
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, buf);

    rszu->addConstant_w_patternSelect(mu,nlp->get_ixu());
    buf = rszu->infnorm_local();
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rszu=%22.17e\n", buf);
  }
  //rsvl = \mu e - sdl * vl
  if(nlp->m_ineq_low()>0) {
    rsvl->setToZero();
    rsvl->axzpy(-1.0, *it.sdl, *it.vl);
    if(nlp->m_ineq_low()<nlp->m_ineq()) rsvl->selectPattern(nlp->get_idl());
    buf = rsvl->infnorm_local();
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, buf);

    //add mu
    rsvl->addConstant_w_patternSelect(mu,nlp->get_idl());
    buf = rsvl->infnorm_local();
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rsvl=%22.17e\n", buf);
  }
  //rsvu = \mu e - sdu * vu
  if(nlp->m_ineq_upp()>0) {
    rsvu->setToZero();
    rsvu->axzpy(-1.0, *it.sdu, *it.vu);

    if(nlp->m_ineq_upp()<nlp->m_ineq()) rsvu->selectPattern(nlp->get_idu());
    buf = rsvu->infnorm_local();
    nrmInf_nlp_complem = fmax(nrmInf_nlp_complem, buf);

    //add mu
    rsvu->addConstant_w_patternSelect(mu,nlp->get_idu());
    buf = rsvu->infnorm_local();
    nrmInf_bar_complem = fmax(nrmInf_bar_complem, buf);
    nlp->log->printf(hovScalars,"NLP resid [update]: inf norm rsvu=%22.17e\n", buf);
  }

#ifdef HIOP_USE_MPI
  //here we reduce each of the norm together for a total cost of 1 Allreduce of 3 doubles
  //otherwise, if calling infnorm() for each vector, there will be 12 Allreduce's, each of 1 double
  double aux[6]={nrmInf_nlp_optim,nrmInf_nlp_feasib,nrmInf_nlp_complem,nrmInf_bar_optim,nrmInf_bar_feasib,nrmInf_bar_complem}, aux_g[6];
  int ierr = MPI_Allreduce(aux, aux_g, 6, MPI_DOUBLE, MPI_MAX, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrmInf_nlp_optim=aux_g[0]; nrmInf_nlp_feasib=aux_g[1]; nrmInf_nlp_complem=aux_g[2];
  nrmInf_bar_optim=aux_g[3]; nrmInf_bar_feasib=aux_g[4]; nrmInf_bar_complem=aux_g[5];
  
  double sum_one_norm = 0.0;
  int ierr = MPI_Allreduce(nrmOne_theta, sum_one_norm, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrmOne_theta = sum_one_norm;
#endif
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
	 nrmInf_nlp_optim, nrmInf_nlp_feasib, nrmInf_nlp_complem);
  printf(" errors (optim/feasib/complem) barrier: %25.16e %25.16e %25.16e\n", 
	 nrmInf_bar_optim, nrmInf_bar_feasib, nrmInf_bar_complem);
}

};

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

// #ifdef HIOP_DEEPCHECKS
//   assert(sxl->allPositive_w_patternSelect(nlp->get_ixl()));
//   assert(sxu->allPositive_w_patternSelect(nlp->get_ixu()));
//   assert(sdl->allPositive_w_patternSelect(nlp->get_idl()));
//   assert(sdu->allPositive_w_patternSelect(nlp->get_idu()));
// #endif
// }
