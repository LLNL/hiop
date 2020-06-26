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

#include "hiopIterate.hpp"

#include <cmath>
#include <cassert>
#include <cstdlib>
namespace hiop
{

hiopIterate::hiopIterate(const hiopNlpFormulation* nlp_)
{
  nlp = nlp_;
  x = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  d = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_ineq_vec());
  sxl = x->alloc_clone();
  sxu = x->alloc_clone();
  sdl = d->alloc_clone();
  sdu = d->alloc_clone();
  //duals
  yc = dynamic_cast<hiopVectorPar*>(nlp->alloc_dual_eq_vec());
  yd = d->alloc_clone();
  zl = x->alloc_clone();
  zu = x->alloc_clone();
  vl = d->alloc_clone();
  vu = d->alloc_clone();
}

hiopIterate::~hiopIterate()
{
  if(x) delete x;
  if(d) delete d;
  if(sxl) delete sxl;
  if(sxu) delete sxu;
  if(sdl) delete sdl;
  if(sdu) delete sdu;
  if(yc) delete yc;
  if(yd) delete yd;
  if(zl) delete zl;
  if(zu) delete zu;
  if(vl) delete vl;
  if(vu) delete vu;
}

/* cloning and copying */
hiopIterate*  hiopIterate::alloc_clone() const
{
  return new hiopIterate(this->nlp);
}


hiopIterate* hiopIterate::new_copy() const
{
  hiopIterate* copy = new hiopIterate(this->nlp);
  copy->copyFrom(*this);
  return copy;
}

void  hiopIterate::copyFrom(const hiopIterate& src)
{
  x->copyFrom(*src.x);
  d->copyFrom(*src.d);

  yc->copyFrom(*src.yc); 
  yd->copyFrom(*src.yd);

  sxl->copyFrom(*src.sxl); 
  sxu->copyFrom(*src.sxu);
  sdl->copyFrom(*src.sdl);
  sdu->copyFrom(*src.sdu);
  zl->copyFrom(*src.zl); 
  zu->copyFrom(*src.zu);
  vl->copyFrom(*src.vl);
  vu->copyFrom(*src.vu);
}

void hiopIterate::print(FILE* f, const char* msg/*=NULL*/) const
{
  if(NULL==msg) fprintf(f, "hiopIterate:\n");
  else fprintf(f, "%s\n", msg);

  x->print(  f, "x:   ");
  d->print(  f, "d:   ");
  yc->print( f, "yc:  "); 
  yd->print( f, "yd:  ");
  sxl->print(f, "sxl: "); 
  sxu->print(f, "sxu: ");
  sdl->print(f, "sdl: ");
  sdu->print(f, "sdu: ");
  zl->print( f, "zl:  "); 
  zu->print( f, "zu:  ");
  vl->print( f, "vl:  ");
  vu->print( f, "vu:  ");
}


void hiopIterate::
projectPrimalsXIntoBounds(double kappa1, double kappa2)
{
  if(!x->projectIntoBounds(nlp->get_xl(),nlp->get_ixl(),
			   nlp->get_xu(),nlp->get_ixu(),
			   kappa1,kappa2)) {
    nlp->log->printf(hovError, "Problem is infeasible due to inconsistent bounds for the variables (lower>upper). Please fix this. In the meanwhile, HiOp will exit (ungracefully).\n");
    exit(-1);
  }
}

void hiopIterate::
projectPrimalsDIntoBounds(double kappa1, double kappa2)
{
  if(!d->projectIntoBounds(nlp->get_dl(),nlp->get_idl(),
		       nlp->get_du(),nlp->get_idu(),
			   kappa1,kappa2)) {
    nlp->log->printf(hovError, "Problem is infeasible due to inconsistent inequality constraints (lower>upper). Please fix this. In the meanwhile, HiOp will exit (ungracefully).\n");
    exit(-1);
  }
}

void hiopIterate::setBoundsDualsToConstant(const double& v)
{
  zl->setToConstant_w_patternSelect(v, nlp->get_ixl());
  zu->setToConstant_w_patternSelect(v, nlp->get_ixu());
  vl->setToConstant_w_patternSelect(v, nlp->get_idl());
  vu->setToConstant_w_patternSelect(v, nlp->get_idu());
}

void hiopIterate::setEqualityDualsToConstant(const double& v)
{
  yc->setToConstant(v);
  yd->setToConstant(v);
}

double hiopIterate::normOneOfBoundDuals() const
{
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  //work locally with all the vectors. This will result in only one MPI_Allreduce call instead of two.
  double nrm1=zl->onenorm_local() + zu->onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr=MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrm1=nrm1_global;
#endif
  nrm1 += vl->onenorm_local() + vu->onenorm_local();
  return nrm1;
}

double hiopIterate::normOneOfEqualityDuals() const
{
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  //work locally with all the vectors. This will result in only one MPI_Allreduce call instead of two.
  double nrm1=zl->onenorm_local() + zu->onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr=MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrm1=nrm1_global;
#endif
  nrm1 += vl->onenorm_local() + vu->onenorm_local() + yc->onenorm_local() + yd->onenorm_local();
  return nrm1;
}

void hiopIterate::normOneOfDuals(double& nrm1Eq, double& nrm1Bnd) const
{
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  //work locally with all the vectors. This will result in only one MPI_Allreduce call
  nrm1Bnd = zl->onenorm_local() + zu->onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr=MPI_Allreduce(&nrm1Bnd, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm());
  assert(MPI_SUCCESS==ierr);
  nrm1Bnd=nrm1_global;
#endif
  nrm1Bnd += vl->onenorm_local() + vu->onenorm_local();
  nrm1Eq   = nrm1Bnd + yc->onenorm_local() + yd->onenorm_local();
}


void hiopIterate::determineSlacks()
{
  sxl->copyFrom(*x);
  sxl->axpy(-1., nlp->get_xl());
  sxl->selectPattern(nlp->get_ixl());

  sxu->copyFrom(nlp->get_xu());
  sxu->axpy(-1., *x); 
  sxu->selectPattern(nlp->get_ixu());

  sdl->copyFrom(*d);
  sdl->axpy(-1., nlp->get_dl());
  sdl->selectPattern(nlp->get_idl());

  sdu->copyFrom(nlp->get_du());
  sdu->axpy(-1., *d); 
  sdu->selectPattern(nlp->get_idu());

#ifdef HIOP_DEEPCHECKS
  assert(sxl->allPositive_w_patternSelect(nlp->get_ixl()));
  assert(sxu->allPositive_w_patternSelect(nlp->get_ixu()));
  assert(sdl->allPositive_w_patternSelect(nlp->get_idl()));
  assert(sdu->allPositive_w_patternSelect(nlp->get_idu()));
#endif
}

void hiopIterate::determineDualsBounds_d(const double& mu)
{
#ifdef DEBUG
  assert(true == sdl->allPositive_w_patternSelect(nlp->get_idl()));
  assert(true == sdu->allPositive_w_patternSelect(nlp->get_idu()));
#endif
  vl->setToConstant(mu);
  vl->componentDiv_w_selectPattern(*sdl, nlp->get_idl());

  vu->setToConstant(mu);
  vu->componentDiv_w_selectPattern(*sdu, nlp->get_idu());
}

bool hiopIterate::
fractionToTheBdry(const hiopIterate& dir, const double& tau, double& alphaprimal, double& alphadual) const
{
  alphaprimal=alphadual=10.0;
  double alpha=0;
  alpha=sxl->fractionToTheBdry_w_pattern(*dir.sxl, tau, nlp->get_ixl());
  alphaprimal=fmin(alphaprimal,alpha);
  
  alpha=sxu->fractionToTheBdry_w_pattern(*dir.sxu, tau, nlp->get_ixu());
  alphaprimal=fmin(alphaprimal,alpha);

  alpha=sdl->fractionToTheBdry_w_pattern(*dir.sdl, tau, nlp->get_idl());
  alphaprimal=fmin(alphaprimal,alpha);

  alpha=sdu->fractionToTheBdry_w_pattern(*dir.sdu, tau, nlp->get_idu());
  alphaprimal=fmin(alphaprimal,alpha);

  //for dual variables
  alpha=zl->fractionToTheBdry_w_pattern(*dir.zl, tau, nlp->get_ixl());
  alphadual=fmin(alphadual,alpha);
  
  alpha=zu->fractionToTheBdry_w_pattern(*dir.zu, tau, nlp->get_ixu());
  alphadual=fmin(alphadual,alpha);

  alpha=vl->fractionToTheBdry_w_pattern(*dir.vl, tau, nlp->get_idl());
  alphadual=fmin(alphadual,alpha);

  alpha=vu->fractionToTheBdry_w_pattern(*dir.vu, tau, nlp->get_idu());
  alphadual=fmin(alphadual,alpha); 
#ifdef HIOP_USE_MPI
  double aux[2]={alphaprimal,alphadual}, aux_g[2];
  int ierr=MPI_Allreduce(aux, aux_g, 2, MPI_DOUBLE, MPI_MIN, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  alphaprimal=aux_g[0]; alphadual=aux_g[1];
#endif

  return true;
}


bool hiopIterate::takeStep_primals(const hiopIterate& iter, const hiopIterate& dir, const double& alphaprimal, const double& alphadual)
{
  x->copyFrom(*iter.x); x->axpy(alphaprimal, *dir.x);
  d->copyFrom(*iter.d); d->axpy(alphaprimal, *dir.d);
  sxl->copyFrom(*iter.sxl); sxl->axpy(alphaprimal,*dir.sxl);
  sxu->copyFrom(*iter.sxu); sxu->axpy(alphaprimal,*dir.sxu);
  sdl->copyFrom(*iter.sdl); sdl->axpy(alphaprimal,*dir.sdl);
  sdu->copyFrom(*iter.sdu); sdu->axpy(alphaprimal,*dir.sdu);
#ifdef HIOP_DEEPCHECKS
  assert(sxl->matchesPattern(nlp->get_ixl()));
  assert(sxu->matchesPattern(nlp->get_ixu()));
  assert(sdl->matchesPattern(nlp->get_idl()));
  assert(sdu->matchesPattern(nlp->get_idu()));
#endif
  return true;
}
bool hiopIterate::takeStep_duals(const hiopIterate& iter, const hiopIterate& dir, const double& alphaprimal, const double& alphadual)
{
  yd->copyFrom(*iter.yd); yd->axpy(alphaprimal, *dir.yd);
  yc->copyFrom(*iter.yc); yc->axpy(alphaprimal, *dir.yc);
  zl->copyFrom(*iter.zl); zl->axpy(alphadual, *dir.zl);
  zu->copyFrom(*iter.zu); zu->axpy(alphadual, *dir.zu);
  vl->copyFrom(*iter.vl); vl->axpy(alphadual, *dir.vl);
  vu->copyFrom(*iter.vu); vu->axpy(alphadual, *dir.vu);
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  return true;
}
/*bool hiopIterate::updateDualsEq(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual)
{
  yc->copyFrom(*iter.yc); yc->axpy(alphaprimal,*dir.yc);
  yd->copyFrom(*iter.yd); yd->axpy(alphaprimal,*dir.yd);
  return true;
}

bool hiopIterate::updateDualsIneq(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual)
{
  zl->copyFrom(*iter.zl); zl->axpy(alphadual,*dir.zl);
  zu->copyFrom(*iter.zu); zu->axpy(alphadual,*dir.zu);
  vl->copyFrom(*iter.vl); vl->axpy(alphadual,*dir.vl);
  vu->copyFrom(*iter.vu); vu->axpy(alphadual,*dir.vu);
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  return true;
}
*/

bool hiopIterate::adjustDuals_primalLogHessian(const double& mu, const double& kappa_Sigma)
{
  zl->adjustDuals_plh(*sxl,nlp->get_ixl(),mu,kappa_Sigma);
  zu->adjustDuals_plh(*sxu,nlp->get_ixu(),mu,kappa_Sigma);
  vl->adjustDuals_plh(*sdl,nlp->get_idl(),mu,kappa_Sigma);
  vu->adjustDuals_plh(*sdu,nlp->get_idu(),mu,kappa_Sigma);
#ifdef HIOP_DEEPCHECKS
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  return true;
}

double hiopIterate::evalLogBarrier() const
{
  double barrier;
  barrier = sxl->logBarrier(nlp->get_ixl());
  barrier+= sxu->logBarrier(nlp->get_ixu());
#ifdef HIOP_USE_MPI
  double res;
  int ierr = MPI_Allreduce(&barrier, &res, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  barrier=res;
#endif
  barrier+= sdl->logBarrier(nlp->get_idl());
  barrier+= sdu->logBarrier(nlp->get_idu());

  return barrier;
}


void  hiopIterate::addLogBarGrad_x(const double& mu, hiopVector& gradx) const
{
  // gradx = grad - mu / sxl = grad - mu * select/sxl
  gradx.addLogBarrierGrad(-mu, *sxl, nlp->get_ixl());
  gradx.addLogBarrierGrad( mu, *sxu, nlp->get_ixu());
}

void  hiopIterate::addLogBarGrad_d(const double& mu, hiopVector& gradd) const
{
  gradd.addLogBarrierGrad(-mu, *sdl, nlp->get_idl());
  gradd.addLogBarrierGrad( mu, *sdu, nlp->get_idu());
}

double hiopIterate::linearDampingTerm(const double& mu, const double& kappa_d) const
{
  double term;
  term  = sxl->linearDampingTerm(nlp->get_ixl(), nlp->get_ixu(), mu, kappa_d);
  term += sxu->linearDampingTerm(nlp->get_ixu(), nlp->get_ixl(), mu, kappa_d);
#ifdef HIOP_USE_MPI
  double res;
  int ierr = MPI_Allreduce(&term, &res, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  term = res;
#endif  
  term += sdl->linearDampingTerm(nlp->get_idl(), nlp->get_idu(), mu, kappa_d);
  term += sdu->linearDampingTerm(nlp->get_idu(), nlp->get_idl(), mu, kappa_d);

  return term;
}

void hiopIterate::addLinearDampingTermToGrad_x(const double& mu, const double& kappa_d, const double& beta, hiopVector& grad_x) const
{
  /*sxl->addLinearDampingTermToGrad(nlp->get_ixl(), nlp->get_ixu(), mu, kappa_d, grad_x);
    sxu->addLinearDampingTermToGrad(nlp->get_ixu(), nlp->get_ixl(), mu, kappa_d, grad_x); */
  //I'll do it in place, in one for loop, to be faster
  const double* ixl=dynamic_cast<const hiopVectorPar&>(nlp->get_ixl()).local_data_const();
  const double* ixu=dynamic_cast<const hiopVectorPar&>(nlp->get_ixu()).local_data_const();
  const double*  xv=x->local_data_const();   long long n_local = x->get_local_size();
  double* gv = dynamic_cast<hiopVectorPar&>(grad_x).local_data();
#ifdef HIOP_DEEPCHECKS
  assert(n_local==dynamic_cast<hiopVectorPar&>(grad_x).get_local_size());
#endif
  
  const double ct=kappa_d*mu;
  if(1.0==beta) {
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(ixl[i]==1.) {
	if(ixu[i]==0.) gv[i] += ct;
      } else {
	if(ixu[i]==1.) gv[i] -= ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(ixl[i]==0); assert(ixu[i]==0);}
#endif
      }
    }
  } else if(-1.==beta) {
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(ixl[i]==1.) {
	if(ixu[i]==0.) gv[i] -= ct;
      } else {
	if(ixu[i]==1.) gv[i] += ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(ixl[i]==0); assert(ixu[i]==0);}
#endif
      }
    }
  } else {
    //beta is neither 1. nor -1.
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(ixl[i]==1.) {
	if(ixu[i]==0.) gv[i] += beta*ct;
      } else {
	if(ixu[i]==1.) gv[i] -= beta*ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(ixl[i]==0); assert(ixu[i]==0);}
#endif
      }
    }
  }
}

void hiopIterate::addLinearDampingTermToGrad_d(const double& mu, const double& kappa_d, const double& beta, hiopVector& grad_d) const
{
  /*sxl->addLinearDampingTermToGrad(nlp->get_ixl(), nlp->get_ixu(), mu, kappa_d, grad_x);
    sxu->addLinearDampingTermToGrad(nlp->get_ixu(), nlp->get_ixl(), mu, kappa_d, grad_x); */
  //I'll do it in place, in one for loop, to be faster
  const double* idl=dynamic_cast<const hiopVectorPar&>(nlp->get_idl()).local_data_const();
  const double* idu=dynamic_cast<const hiopVectorPar&>(nlp->get_idu()).local_data_const();
  const double*  dv=d->local_data_const();   long long n_local = d->get_local_size();
  double* gv = dynamic_cast<hiopVectorPar&>(grad_d).local_data();
#ifdef HIOP_DEEPCHECKS
  assert(n_local==dynamic_cast<hiopVectorPar&>(grad_d).get_local_size());
#endif
  
  const double ct=kappa_d*mu;
  if(beta==-1.0) {
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(idl[i]==1.) {
	if(idu[i]==0.) gv[i] -= ct;
      } else {
	if(idu[i]==1.) gv[i] += ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(idl[i]==0); assert(idu[i]==0);}
#endif
      }
    }
  } else if(1.0==beta) {
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(idl[i]==1.) {
	if(idu[i]==0.) gv[i] += ct;
      } else {
	if(idu[i]==1.) gv[i] -= ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(idl[i]==0); assert(idu[i]==0);}
#endif
      }
    }
  } else {
    //beta is neither 1. nor -1.
    for(long i=0; i<n_local; i++) {
      //!opt -> can be streamlined with blas:  gv[i] += ct*(ixl[i]-ixu[i]);
      if(idl[i]==1.) {
	if(idu[i]==0.) gv[i] += beta*ct;
      } else {
	if(idu[i]==1.) gv[i] -= beta*ct;
#ifdef HIOP_DEEPCHECKS
	else {assert(idl[i]==0); assert(idu[i]==0);}
#endif
      }
    }
  }
}

};
