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

void hiopInnerProdMatrixFreeWeight::apply(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x)
{
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
  nlp->applyH(x.local_data());
}
// y = beta*y + alpha*This^{-1}*x
void hiopInnerProdMatrixFreeWeight::applyInverse(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x)
{
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
  nlp->applyHInv(x.local_data());
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


} //end of namespace
