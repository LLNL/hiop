#ifndef HIOP_FACT_ACCEPTOR
#define HIOP_FACT_ACCEPTOR

#include "hiopNlpFormulation.hpp"
#include "hiopPDPerturbation.hpp"

namespace hiop
{

class hiopFactAcceptor
{
public:
  /** Default constructor 
   * Determine if factorization is accepted or not
   */
  hiopFactAcceptor(hiopPDPerturbation* p)
  : perturb_calc_{p}
  {}

  virtual ~hiopFactAcceptor() 
  {}
  
  /** 
   * Returns '1' if current factorization is rejected and need re-factorize the matrix
   * Returns '0' if current factorization is ok
   * Returns '-1' if current factorization failed due to singularity
   */
  virtual int requireReFactorization(const hiopNlpFormulation& nlp, const int& n_neg_eig,
                                     double& delta_wx, double& delta_wd, double& delta_cc, double& delta_cd) = 0;
  
  /** Set log-barrier mu. */
  inline void set_mu(const double& mu)
  {
    perturb_calc_->set_mu(mu);
  }
      
protected:  
  hiopPDPerturbation* perturb_calc_;
  
};
  
class hiopFactAcceptorIC : public hiopFactAcceptor
{
public:
  /** Default constructor 
   * Determine if factorization is accepted or not
   */
  hiopFactAcceptorIC(hiopPDPerturbation* p, const long long n_required_neg_eig)
  : hiopFactAcceptor(p),
    n_required_neg_eig_{n_required_neg_eig}
  {}

  virtual ~hiopFactAcceptorIC() 
  {}
   
  virtual int requireReFactorization(const hiopNlpFormulation& nlp, const int& n_neg_eig, 
                                     double& delta_wx, double& delta_wd, double& delta_cc, double& delta_cd);
 
protected:
  int n_required_neg_eig_;    
};
  
} //end of namespace
#endif
