#include "hiopInnerProdWeight.hpp"


/******* Implementation for class hiopInnerProdMatrixFreeWeight : public hiopInnerProdWeight */

namespace hiop
{

hiopInnerProdMatrixFreeWeight::
hiopInnerProdMatrixFreeWeight(hiopNlpFormulation* nlp_) 
  : nlp(nlp_),  hiopInnerProdWeight() 
{
  vec_aux = dynamic_cast<hiopVectorPar*>(nlp_->alloc_primal_vec());
};

hiopInnerProdMatrixFreeWeight::~hiopInnerProdMatrixFreeWeight() 
{
  delete vec_aux;
};

void hiopInnerProdMatrixFreeWeight::apply(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x)
{
  assert(false);
  if(1.0!=beta) 
    y.scale(beta);

  if(0.0!=alpha) {
    vec_aux->copyFrom(x);
    nlp->applyH(vec_aux->local_data());
    y.axpy(alpha,*vec_aux);
  }
}
 
// x = This*x
void hiopInnerProdMatrixFreeWeight::apply(hiopVectorPar& x)
{
  //!assert(false);
  nlp->applyH(x.local_data());
}

/* return the Lebesque measure of the domain Omega */
double hiopInnerProdMatrixFreeWeight::totalVolume()
{
  double vol;
  vec_aux->setToConstant(1.);
  nlp->applyH(vec_aux->local_data());
  //take one norm which is  computing the sum of the elements in this case (since the elems are positive)
  return vec_aux->onenorm();
}

// y = beta*y + alpha*This^{-1}*x
void hiopInnerProdMatrixFreeWeight::applyInverse(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x)
{

  assert(false);

  if(1.0!=beta) 
    y.scale(beta);

  if(0.0!=alpha) {
    vec_aux->copyFrom(x);
    nlp->applyHInv(vec_aux->local_data());
    y.axpy(alpha,*vec_aux);
  }
}
// x = This^{-1}*x
void hiopInnerProdMatrixFreeWeight::applyInverse(hiopVectorPar& x)
{
  assert(false);
  nlp->applyHInv(x.local_data());
}


void hiopInnerProdMatrixFreeWeight::
applyAdjoint(const double& beta, hiopVectorPar& y, const hiopMatrix& Ja, const double& alpha, const hiopVectorPar& x)
{
  assert(false);
  if(1.0!=beta) y.scale(beta);

  if(0.0!=alpha) {
    Ja.transTimesVec(0.0, *vec_aux, 1.0, x);
    nlp->applyHInv(vec_aux->local_data());
    y.axpy(alpha,*vec_aux);
  }
}


//x'*This*x
double hiopInnerProdMatrixFreeWeight::primalnorm(const hiopVectorPar& x)
{
  vec_aux->copyFrom(x);
  nlp->applyH(vec_aux->local_data());

  return sqrt(vec_aux->dotProductWith(x));
}
  //x'*This^{-1}*x
double hiopInnerProdMatrixFreeWeight::dualnorm(const hiopVectorPar& x)
{
  assert(false);
  vec_aux->copyFrom(x);
  nlp->applyHInv(vec_aux->local_data());
  return sqrt(vec_aux->dotProductWith(x));
}

double hiopInnerProdMatrixFreeWeight::dotProd(const hiopVectorPar& u, const hiopVectorPar& v)
{

  vec_aux->copyFrom(u);
  nlp->applyH(vec_aux->local_data());
  return vec_aux->dotProductWith(v);
}

//!!! this is a "local" function: the code calling this function is responsible for MPI_sum_reducing !!!
double hiopInnerProdMatrixFreeWeight::logBarrier(const hiopVectorPar& d, const hiopVectorPar& id)
{

  const double* dd = d.local_data_const();
  const double* ii = id.local_data_const();

  vec_aux->setToConstant(1.0);
  nlp->applyH(vec_aux->local_data());

  double* mm = vec_aux->local_data();

  const long long loc_n = d.get_local_size();
#ifdef DEEP_CHECKING
  bool allNonzero=true;
#endif
  double logbar=0.0;
  for(int i=0; i<loc_n; i++) {
    logbar += mm[i]*log(dd[i])*ii[i];
#ifdef DEEP_CHECKING
    allNonzero = allNonzero && (ii[i]==1.0);
    //if(ii[i]!=1.) printf("-- %d %d\n", i, allNonzero);
    //printf (" %g ", mm[i]);
#endif
  }
#ifdef DEEP_CHECKING
  //printf (" \n ");
  assert(true==allNonzero && "This function should be used for vectors with complete/full nonzero pattern.");
#endif
  return logbar;
}

void hiopInnerProdMatrixFreeWeight::addLogBarGrad(const hiopVectorPar&x, const hiopVector& ix, 
						  const double& mu, 
						  hiopVectorPar& grad)
{
  assert(false);
#ifdef DEEP_CHECKING
  vec_aux->setToConstant(1.);
  bool allone = ix.matchesPattern(*vec_aux);
  vec_aux->setToZero();
  bool allzero= ix.matchesPattern(*vec_aux);
  assert ( (allzero||allone) && "Nonzero pattern of the argument should be completely zero or completely nonzero");
#endif
  //gradx = gradx + mu* M* [[ 1/x ]]
  vec_aux->setToConstant(1.0); 
  vec_aux->componentDiv_p_selectPattern(x,ix);
  nlp->applyH(vec_aux->local_data());
  grad.axpy(mu, *vec_aux);
}

//!!! this is a "local" function: the code calling this function is responsible for MPI_sum_reducing !!!
double hiopInnerProdMatrixFreeWeight::linearDampingTerm(const hiopVectorPar& d, 
						      const hiopVectorPar& dleft, 
						      const hiopVectorPar& dright,
						      const double& mu, const double& kappa_d)
{

#ifdef DEEP_CHECKING
  //check dleft all zero || dleft all one 
  bool ok=true;
  vec_aux->setToConstant(1.0);
  bool allone=d.matchesPattern(*vec_aux);
  
  vec_aux->setToZero();
  bool allzero=d.matchesPattern(*vec_aux);
  assert(true == (allone || allzero));
#endif

  vec_aux->copyFrom(d);
  nlp->applyH(vec_aux->local_data());
  
  return vec_aux->linearDampingTerm(dleft,dright,mu,kappa_d);
}

} //end of namespace
