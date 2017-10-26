#ifndef HIOP_INNERPRODWEIGHT
#define HIOP_INNERPRODWEIGHT

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"

#include <cassert>

namespace hiop
{

  /** Abstract class for the weight matrix that appears from discretizaton of 
   * infinite dimensional inner products.
   * 
   * Default implementation is for the finite dimensional case, that is the weight
   * operator is identity
   */
  class hiopInnerProdWeight
  {
  public:
    hiopInnerProdWeight() {};
    virtual ~hiopInnerProdWeight() {};
    
    //apply the weight matrix
    // y = beta*y + alpha*This*x; by default, This=I. One should not really call this method
    virtual void apply(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x)
    {
      if(1.!=beta)  y.scale(beta);
      if(0.!=alpha) y.axpy(alpha,x);
    };
    // x = This*x; by default applies the identity, that is, it does nothing
    virtual void apply(hiopVector& x) { };
    // y = beta*y + alpha*This^{-1}*x; by default, This=I. One should not really call this
    virtual void applyInverse(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x)
    {
      if(1.!=beta)  y.scale(beta);
      if(0.!=alpha) y.axpy(alpha,x);
    }
    // x = This^{-1}*x; by default applies the identity, that is, it does nothing
    virtual void applyInverse(hiopVector& x) {};

    virtual double norm(const hiopVectorPar& x) {return x.twonorm();}
    virtual double dotProd(const hiopVectorPar& u, const hiopVectorPar& v) { return u.dotProductWith(v); }
    virtual bool identityAsVec(hiopVectorPar& d) { assert(false && "not yet implemented."); return true;}
  };
  
  /** Matrix free weight matrix in the sense that it does not store the matrix */
  class hiopInnerProdMatrixFreeWeight : public hiopInnerProdWeight
  {
  public: 
    hiopInnerProdMatrixFreeWeight(hiopNlpFormulation* nlp_);
    virtual ~hiopInnerProdMatrixFreeWeight();

    // y = beta*y + alpha*This*x; by default, This=I. One should not really call this method
    virtual void apply(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x);
   // x = This*x
    virtual void apply(hiopVectorPar& x);
    // y = beta*y + alpha*This^{-1}*x
    virtual void applyInverse(double beta, hiopVectorPar& y, double alpha, const hiopVectorPar& x);
    // x = This^{-1}*x
    virtual void applyInverse(hiopVectorPar& x);

    virtual double norm(const hiopVectorPar& x);
    virtual double dotProd(const hiopVectorPar& u, const hiopVectorPar& v); 
    virtual bool identityAsVec(hiopVectorPar& d) { assert(false && "not yet implemented."); return true;}
  protected:
    hiopNlpFormulation* nlp;
    //we need an auxiliary vector
    hiopVectorPar* vec_aux;
  private: 
    hiopInnerProdMatrixFreeWeight() : nlp(NULL), vec_aux(NULL) { assert(false); }
  };

  /** Diagonal "mass" matrix */
  class hiopInnerProdDiagonalMatrixWeight : public hiopInnerProdWeight
  {
  public: 
    hiopInnerProdDiagonalMatrixWeight(hiopNlpFormulation* nlp_) :  hiopInnerProdWeight() 
    {
      diag_mat = dynamic_cast<hiopVectorPar*>(nlp_->alloc_primal_vec());
      assert(NULL==diag_mat);
      //nlp->get_mass_matrix(*diag_mat);
      assert(false && "to be implemented");
    };
    virtual ~hiopInnerProdDiagonalMatrixWeight() 
    { delete diag_mat; };
  protected:
    hiopVectorPar* diag_mat;
  private: 
    hiopInnerProdDiagonalMatrixWeight() : diag_mat(NULL) { assert(false); }
    hiopInnerProdDiagonalMatrixWeight(const hiopInnerProdDiagonalMatrixWeight& other) { assert(false); }
  };

} //end of namespace

#endif
