#ifndef HIOP_INNERPRODWEIGHT
#define HIOP_INNERPRODWEIGHT

//#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"

#include <cassert>

namespace hiop
{

  /** Abstract class for the weight matrix that appears from discretizaton of 
   * infinite dimensional inner products
   */
  class hiopInnerProdWeight
  {
  public:
    hiopInnerProdWeight() {};
    virtual ~hiopInnerProdWeight() {};
    
    //apply the weight matrix
    virtual bool apply(double* x) {};
    virtual bool applyInverse(double* x) {};
  };
  
  /** Matrix free weight matrix in the sense that it does not store the matrix */
  class hiopInnerProdMatrixFreeWeight : public hiopInnerProdWeight
  {
  public: 
    hiopInnerProdMatrixFreeWeight(hiopNlpFormulation* nlp_) : nlp(nlp_),  hiopInnerProdWeight() {};
    virtual ~hiopInnerProdMatrixFreeWeight() {};
    virtual bool apply(double* x)        { return nlp->applyH(x); };
    virtual bool applyInverse(double* x) { return nlp->applyHInv(x); };
  protected:
    hiopNlpFormulation* nlp;
  private: 
    hiopInnerProdMatrixFreeWeight() : nlp(NULL) { assert(false); }
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
