#ifndef HIOP_REDIDUAL_AUGLAGR_HPP
#define HIOP_REDIDUAL_AUGLAGR_HPP

#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopVector.hpp"


namespace hiop
{

class hiopResidualAugLagr
{
public:
  hiopResidualAugLagr(hiopAugLagrNlpAdapter *nlp_in, long long n_vars, long long m_constraints) :
    nlp(nlp_in),
    _penaltyFcn(new hiopVectorPar(m_constraints)),
    _grad(new hiopVectorPar(n_vars)),
    _nrmInfOptim(-1.),
    _nrmInfFeasib(-1.) {};
  
  ~hiopResidualAugLagr()
  {
      delete _penaltyFcn;
      delete _grad;
  }
  
  //danger!!! accessor to private data, can make our object incnsistent,
  //user needs to call update() to recompute the norms after changing the
  //residual vectors
  //We pass the pointer directly to the AugLagrAdapter in order to avoid
  //the additional copy of the residual vectors in-between
  inline double *getFeasibilityPtr() {return _penaltyFcn->local_data(); }
  inline double *getOptimalityPtr()  {return _grad->local_data(); }

  void update()
  { 
    //update scaled norm of the duals
    double sd = 1.;
    //nlp->get_dualScaling(sd);
    _nrmInfOptim = _grad->infnorm() / sd;

    //update feasibility norm
    _nrmInfFeasib = _penaltyFcn->infnorm();
  }

  /* Return the Nlp errors computed at the previous update call. */ 
  inline double getFeasibilityNorm() const {return _nrmInfFeasib; }
  inline double getOptimalityNorm()  const {return _nrmInfOptim; }
  inline void getNlpErrors(double& optim, double& feas) const
  { optim=_nrmInfOptim; feas=_nrmInfFeasib;};

  /* residual printing function - calls hiopVector::print 
   * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* msg=NULL, int max_elems=-1, int rank=-1) const;

private:
  hiopAugLagrNlpAdapter *nlp; ///< representation of the problem
  hiopVectorPar *_penaltyFcn; ///< penalty function p(x,s) residuals
  hiopVectorPar *_grad;   ///< gradient; the first KKT optimality condition

  double _nrmInfOptim;  // norm of the gradient
  double _nrmInfFeasib;  // norm of the penalty function
  
private:
  hiopResidualAugLagr() {};
  hiopResidualAugLagr(const hiopResidualAugLagr&) {};
  hiopResidualAugLagr& operator=(const hiopResidualAugLagr& o) {return *this;};
};

}
#endif
