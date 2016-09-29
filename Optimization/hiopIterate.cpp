#include "hiopIterate.hpp"

#include <cmath>
#include <cassert>

hiopIterate::hiopIterate(const hiopNlpDenseConstraints* nlp_)
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

/*
hiopIterate* hiopIterate::new_copy() const
{
  hiopIterate* copy = new hiopIterate(this->nlp);
  copy->x->copyFrom(*this->x);
  }
*/
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

void hiopIterate::print()
{
  printf("x: "); x->print();
  printf("d: "); d->print();
  printf("yc: "); yc->print(); 
  printf("yd: "); yd->print();

  printf("sxl: "); sxl->print(); 
  printf("sxu: "); sxu->print();
  printf("sdl: "); sdl->print();
  printf("sdu: "); sdu->print();
  printf("zl: "); zl->print(); 
  printf("zu: "); zu->print();
  printf("vl: "); vl->print();
  printf("vu: "); vu->print();
}


void hiopIterate::
projectPrimalsXIntoBounds(double kappa1, double kappa2)
{
  x->projectIntoBounds(nlp->get_xl(),nlp->get_ixl(),
		       nlp->get_xu(),nlp->get_ixu(),
		       kappa1,kappa2);
}

void hiopIterate::
projectPrimalsDIntoBounds(double kappa1, double kappa2)
{
  d->projectIntoBounds(nlp->get_dl(),nlp->get_idl(),
		       nlp->get_du(),nlp->get_idu(),
		       kappa1,kappa2);
}


void hiopIterate::setBoundsDualsToConstant(const double& v)
{
  zl->setToConstant_w_patternSelect(v, nlp->get_ixl());
  zu->setToConstant_w_patternSelect(v, nlp->get_ixu());
  vl->setToConstant_w_patternSelect(v, nlp->get_idl());
  vu->setToConstant_w_patternSelect(v, nlp->get_idu());
#ifdef WITH_GPU
  //maybe do the above arithmetically zl->setToConstant(); zl=zl.*ixl
#endif
}

void hiopIterate::setEqualityDualsToConstant(const double& v)
{
  yc->setToConstant(v);
  yd->setToConstant(v);
}


double hiopIterate::normOneOfBoundDuals() const
{
  return 0.0;
}

double hiopIterate::normOneOfEqualityDuals() const
{
#ifdef DEEP_CHECKING
  assert(zl->matchesPattern(nlp->get_ixl()));
  assert(zu->matchesPattern(nlp->get_ixu()));
  assert(vl->matchesPattern(nlp->get_idl()));
  assert(vu->matchesPattern(nlp->get_idu()));
#endif
  //work locally with all the vectors. This will result in only one MPI_Allreduce call instead of fours.
  double nrm1=zl->onenorm_local() + zu->onenorm_local() + vl->onenorm_local() + vu->onenorm_local();
#ifdef WITH_MPI
  double nrm1_global;
  int ierr=MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(MPI_SUCCESS==ierr);
  nrm1=nrm1_global;
#endif
  return nrm1;
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

#ifdef DEEP_CHECKING
  assert(sxl->allPositive_w_patternSelect(nlp->get_ixl()));
  assert(sxu->allPositive_w_patternSelect(nlp->get_ixu()));
  assert(sdl->allPositive_w_patternSelect(nlp->get_idl()));
  assert(sdu->allPositive_w_patternSelect(nlp->get_idu()));
#endif
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
  return true;
}


bool hiopIterate::updatePrimals(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual)
{
  x->copyFrom(*iter.x); x->axpy(alphaprimal, *dir.x);
  d->copyFrom(*iter.d); d->axpy(alphaprimal, *dir.d);
  sxl->copyFrom(*iter.sxl); sxl->axpy(alphaprimal,*dir.sxl);
  sxu->copyFrom(*iter.sxu); sxu->axpy(alphaprimal,*dir.sxu);
  sdl->copyFrom(*iter.sdl); sdl->axpy(alphaprimal,*dir.sdl);
  sdu->copyFrom(*iter.sdu); sdu->axpy(alphaprimal,*dir.sdu);
#ifdef DEEP_CHECKING
  assert(sxl->matchesPattern(nlp->get_ixl()));
  assert(sxu->matchesPattern(nlp->get_ixu()));
  assert(sdl->matchesPattern(nlp->get_idl()));
  assert(sdu->matchesPattern(nlp->get_idu()));
  //  assert(dir->zl->matchesPattern(nlp->get_ixl()));
  //assert(dir->zu->matchesPattern(nlp->get_ixu()));
  //assert(dir->vl->matchesPattern(nlp->get_idl()));
  //assert(dir->vu->matchesPattern(nlp->get_idu()));
#endif
  return true;
}

bool hiopIterate::updateDualsEq(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual)
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
#ifdef DEEP_CHECKING
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
  barrier+= sdl->logBarrier(nlp->get_idl());
  barrier+= sdu->logBarrier(nlp->get_idu());

  return barrier;
}
