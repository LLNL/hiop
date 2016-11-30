#ifndef HIOP_LOGBARRPROB
#define HIOP_LOGBARRPROB

class hiopLogBarProblem
{
public:
  hiopLogBarProblem(hiopNlpDenseConstraints* nlp_) : nlp(nlp_), kappa_d(1e-5) {};

  double mu;
  //just a proxy: keeps pointers to the problem's data and updates LogBar func, grad and all that on the fly
  const hiopVector *c,*d;
  const hiopVector *grad_f;
  const hiopMatrixDense* Jac_c;
  const hiopMatrixDense* Jac_d;

  //algorithm's parameters 
  // factor in computing the linear damping terms used to control unboundness in the log-barrier problem (Section 3.7) */
  double kappa_d;      
public:
  //update with the NLP problem data given by the parameters
  inline void 
  updateWithNlpInfo(const hiopIterate& iter, const double& mu_, 
		const double &f, const hiopVector& c_, const hiopVector& d_, 
		const hiopVector& gradf_,  const hiopMatrixDense& Jac_c_,  const hiopMatrixDense& Jac_d_) 
  {
    mu=mu_; _f=f; c=&c_; d=&d_; grad_f=&gradf_; Jac_c=&Jac_c_; Jac_d=&Jac_d_; _iter=&iter;
  }
  /* gradx += beta*grad_x_logBar*/
  inline void addLogBarTermsToGrad_x(const double& beta, hiopVector& gradx) const
  {
    if(kappa_d>0.) _iter->addLinearDampingTermToGrad_x(mu,kappa_d,beta,gradx);
  }
  /* gradd += beta*grad_d_logBar*/
  inline void addLogBarTermsToGrad_d(const double& beta, hiopVector& gradd) const
  {
    if(kappa_d>0.) _iter->addLinearDampingTermToGrad_d(mu,kappa_d,beta,gradd);
  }
  inline double f_logBar()
  {
    double f_log = mu * _iter->evalLogBarrier();
    if(kappa_d>0.) 
      f_log += _iter->linearDampingTerm(mu,kappa_d);
    f_log += _f;
    return f_log;
  }


protected:
  const hiopIterate* _iter;
  double _f;
  hiopNlpDenseConstraints* nlp;
private:
  hiopLogBarProblem() {};
  hiopLogBarProblem(const hiopLogBarProblem&) {};
  hiopLogBarProblem& operator=(const hiopLogBarProblem&) {return *this;};
};

#endif
