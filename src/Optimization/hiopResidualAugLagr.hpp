#ifndef HIOP_REDIDUAL_AUGLAGR_HPP
#define HIOP_REDIDUAL_AUGLAGR_HPP

#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopVector.hpp"


namespace hiop
{

class hiopResidualAugLagr
{
public:
  hiopResidualAugLagr(long long n_vars, long long m_constraints) :
    _penaltyFcn(new hiopVectorPar(m_constraints)),
    _gradLagr(new hiopVectorPar(n_vars)),
    _nrmInfOptim(-1.),
    _nrmInfFeasib(-1.) {};
  
  ~hiopResidualAugLagr()
  {
      delete _penaltyFcn;
      delete _gradLagr;
  }
  
  //danger!!! accessor to private data, can make our object incnsistent,
  //user needs to call update() to recompute the norms after changing the
  //residual vectors
  //We pass the pointer directly to the AugLagrAdapter in order to avoid
  //the additional copy of the residual vectors in-between
  inline double *getFeasibilityPtr() {return _penaltyFcn->local_data(); }
  inline double *getOptimalityPtr()  {return _gradLagr->local_data(); }

  void update()
  { _nrmInfOptim = _gradLagr->infnorm();
    _nrmInfFeasib = _penaltyFcn->infnorm(); }

  /* Return the Nlp errors computed at the previous update call. */ 
  inline double getFeasibilityNorm() const {return _nrmInfFeasib; }
  inline double getOptimalityNorm()  const {return _nrmInfOptim; }
  inline void getNlpErrors(double& optim, double& feas) const
  { optim=_nrmInfOptim; feas=_nrmInfFeasib;};

  /* residual printing function - calls hiopVector::print 
   * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* msg=NULL, int max_elems=-1, int rank=-1) const;

private:
  hiopVectorPar *_penaltyFcn; ///< penalty function p(x,s) residuals
  hiopVectorPar *_gradLagr;   ///< gradient of the Lagrangian d_L = d_f + J^T lam

  double _nrmInfOptim;  // norm of the gradient of the Lagrangian 
  double _nrmInfFeasib;  // norm of the penalty function
  
private:
  hiopResidualAugLagr() {};
  hiopResidualAugLagr(const hiopResidualAugLagr&) {};
  hiopResidualAugLagr& operator=(const hiopResidualAugLagr& o) {return *this;};
};

}
#endif
