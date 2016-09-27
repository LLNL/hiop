#ifndef HIOP_ALGFilterIPM
#define HIOP_ALGFilterIPM

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopFilter.hpp"
#include "hiopHessianLowRank.hpp"

class hiopAlgFilterIPM
{
public:
  hiopAlgFilterIPM(hiopNlpDenseConstraints* nlp);
  virtual ~hiopAlgFilterIPM();

  virtual int run();
  virtual int defaultStartingPoint(hiopIterate& it_ini);

  /* internal helpers related to the error computation. call NlpAndLogBarrierErrors whenever both 
   * both NLP and Log-Barrier are to be computed, since it is faster than two separate calls */
  virtual bool nlpAndLogBarrierErrors(const hiopIterate& it, 
				      const hiopResidual& resid, 
				      const double& mu, 
				      double& err_nlp, double& err_log);
  virtual double nlpError(const hiopIterate& it, const hiopResidual& resid);
  virtual double barrierError(const hiopIterate& it, const hiopResidual& resid, const double& mu);
private:
  virtual bool updateLogBarrierParameters(const hiopIterate& it, const double& mu_curr, const double& tau_curr,
					  double& mu_new, double& tau_new);
  virtual bool updateLogBarrierProblem(hiopIterate& it, double mu, double &f,  hiopVector& c, hiopVector& d, 
				       hiopVector& grad,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d);
  virtual double thetaLogBarrier(const hiopIterate& it, const hiopResidual& resid, const double& mu);

private:
  hiopNlpDenseConstraints* nlp;
  hiopFilter filter;

  /* Iterate, search directions (managed by the algorithm) */
  hiopIterate*it_curr;
  hiopIterate*it_trial;
  hiopIterate* dir;

  hiopResidual* resid, *resid_trial;

  /* Log-barrier problem data 
   *  The algorithm manages these and updates them by calling the   
   *  problem formulation and then adding the contribution from the 
   *  log-barrier term(s). The data that is not iterate dependent,  
   *  such as lower or upper bounds, is in the NlpFormulation       
   */
  double _f, _f_trial;
  hiopVector *_c,*_d, *_c_trial, *_d_trial;
  hiopVector* _grad_f, *_grad_f_trial; //gradient of the log-barrier objective function
  hiopMatrixDense* _Jac_c, *_Jac_c_trial; //Jacobian of c(x), the equality part
  hiopMatrixDense* _Jac_d, *_Jac_d_trial; //Jacobian of d(x), the inequality part

  hiopHessianInvLowRank* _Hess;

  /** Algorithms's working quantities */  
  double mu, tau;
  //initialized to 1e4*max{1,\theta(x_0)} and used in the filter as an upper acceptability limit for infeasibility
  double theta_max; 
  //1e-4*max{1,\theta(x_0)} used in the switching condition during the line search
  double theta_min;
  /*** Algorithm's parameters ***/
  double mu0;          //intial mu
  double kappa_mu;     //linear decrease factor in mu 
  double theta_mu;     //exponent for a Mehtrotra-style decrease of mu
  double eps_tol;      //solving tolerance for the NLP error
  double tau_min;      //min value for the fraction-to-the-boundary parameter: tau_k=max{tau_min,1-\mu_k}
  double kappa_eps;    //tolerance for the barrier problem, relative to mu: error<=kappa_eps*mu
  double kappa1,kappa2;//params for default starting point
  double smax;         //threshold for the magnitude of the multipliers used in the error estimation
private:
  hiopAlgFilterIPM() {};
  hiopAlgFilterIPM(const hiopAlgFilterIPM& ) {};
  hiopAlgFilterIPM& operator=(const hiopAlgFilterIPM&) {};
};

#endif
