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
    virtual void apply(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x)
    {
      if(1.!=beta)  y.scale(beta);
      if(0.!=alpha) y.axpy(alpha,x);
    };
    // x = This*x; by default applies the identity, that is, it does nothing
    virtual void apply(hiopVectorPar& x) { };
    // y = beta*y + alpha*This^{-1}*x; by default, This=I. One should not really call this
    virtual void applyInverse(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x)
    {
      if(1.!=beta)  y.scale(beta);
      if(0.!=alpha) y.axpy(alpha,x);
    }
    // x = This^{-1}*x; by default applies the identity, that is, it does nothing
    virtual void applyInverse(hiopVectorPar& x) {};

    virtual double primalnorm(const hiopVectorPar& x) {return x.twonorm();}
    virtual double dualnorm(const hiopVectorPar& x) {return x.twonorm();}
    virtual double dotProd(const hiopVectorPar& u, const hiopVectorPar& v) { return u.dotProductWith(v); }
    virtual bool identityAsVec(hiopVectorPar& d) { assert(false && "not yet implemented."); return true;}

    /* applies an adjoint operator given by a finite dimensional matrix approximation 
     * y = beta*y + alpha * Da^* * x;  Da^* is R^{-1}*Ja^T
     */
    virtual void applyAdjoint(const double& beta, hiopVectorPar& y, 
			      const hiopMatrix& Ja, const double& alpha, const hiopVectorPar& x)
    {
      Ja.transTimesVec(beta, y, alpha, x); //return true;
    }

    //compute integral d(t) dt = e'* H * d; considers only entries i such that id[i]==1.
    //!!! this is a "local" function: the code calling this function is responsible for sum_reducing !!!
    virtual double logBarrier(const hiopVectorPar& d, const hiopVectorPar& id) { return d.logBarrier(id); }

    //gradient of the logBarrier (not the derivative) 
    virtual void addLogBarGrad(const hiopVectorPar&x, const hiopVector& ix, const double& mu, hiopVectorPar& grad) 
    {
      grad.addLogBarrierGrad( mu, x, ix);
    };
    // add linear damping terms for d (sxl or sxu) given nonzero pattern/select left and right 
    //selects (ixl and ixu or ixu and ixl).
    // !!! "local" function: the code calling this function is responsible for sum_reducing !!!
    virtual double linearDampingTerm(const hiopVectorPar& d, const hiopVectorPar& dleft, const hiopVectorPar& dright,
				   const double& mu, const double& kappa_d)
    {
      return d.linearDampingTerm(dleft,dright,mu,kappa_d);
    };
    /* perform l2 to L2 or H1 transformations of derivatives if needed */
    virtual void transformGradient(double* grad) { };
    /* Jac_c_L2 from  Jac_c_l2, usually is done by right multiply with inv(M) */
    virtual void transformJacobian(const long long& m, const long long& n, double**Jac_c){ };

  };
  
  /** Matrix free weight matrix in the sense that it does not store the matrix */
  class hiopInnerProdMatrixFreeWeight : public hiopInnerProdWeight
  {
  public: 
    hiopInnerProdMatrixFreeWeight(hiopNlpFormulation* nlp_);
    virtual ~hiopInnerProdMatrixFreeWeight();

    // y = beta*y + alpha*This*x; by default, This=I. One should not really call this method
    virtual void apply(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x);
   // x = This*x
    virtual void apply(hiopVectorPar& x);
    // y = beta*y + alpha*This^{-1}*x
    virtual void applyInverse(const double& beta, hiopVectorPar& y, const double& alpha, const hiopVectorPar& x);
    // x = This^{-1}*x
    virtual void applyInverse(hiopVectorPar& x);

    virtual void applyAdjoint(const double& beta, hiopVectorPar& y, 
			      const hiopMatrix& Ja, const double& alpha, const hiopVectorPar& x);

    virtual double primalnorm(const hiopVectorPar& x);
    virtual double dualnorm(const hiopVectorPar& x);
    virtual double dotProd(const hiopVectorPar& u, const hiopVectorPar& v); 
    virtual bool identityAsVec(hiopVectorPar& d) { assert(false && "not yet implemented."); return true;}
    //compute integral d(t) dt = e'* H * d; considers only entries i such that id[i]==1
    //!!! this is a "local" function: the code calling this function is responsible for sum_reducing !!!
    virtual double logBarrier(const hiopVectorPar& d, const hiopVectorPar& id);
    virtual void addLogBarGrad(const hiopVectorPar&x, const hiopVector& ix, const double& mu, hiopVectorPar& grad);
    virtual double linearDampingTerm(const hiopVectorPar& d, const hiopVectorPar& dleft, const hiopVectorPar& dright,
			   const double& mu, const double& kappa_d);

    /* we work for now under the assumption that the problem has l2 specification */
    virtual void transformGradient(double* grad) { nlp->applyHInv(grad); };
    virtual void transformJacobian(const long long& m, const long long& n, double**Jac) 
    {
      for(int i=0; i<m; i++)
	nlp->applyHInv(Jac[i]);
    };
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
