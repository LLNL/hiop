#ifndef HIOP_LOGBARRPROB
#define HIOP_LOGBARRPROB

class hiopLogBarProblem
{
public:
  hiopLogBarProblem(hiopNlpDenseConstraints* nlp_) : nlp(nlp_), kappa_d(1e-5) 
  {
    _grad_x_logbar = nlp->alloc_primal_vec();
    _grad_d_logbar = nlp->alloc_dual_ineq_vec();
  };
  virtual ~hiopLogBarProblem()
  {
    delete _grad_x_logbar;
    delete _grad_d_logbar;
  };
public: //members
  double mu;
  double f_logbar, f_logbar_trial;
  hiopVector *_grad_x_logbar, *_grad_d_logbar; //of the log barrier
  //just proxies: keeps pointers to the problem's data and updates LogBar func, grad and all that on the fly
  const hiopIterate *iter, *iter_trial;
  const hiopVector *c_nlp,*d_nlp, *c_nlp_trial, *d_nlp_trial;
  const hiopMatrixDense *Jac_c_nlp, *Jac_d_nlp;

    //algorithm's parameters 
  // factor in computing the linear damping terms used to control unboundness in the log-barrier problem (Section 3.7) */
  double kappa_d;      
public:
  //update with the NLP problem data given by the parameters
  inline void 
  updateWithNlpInfo(const hiopIterate& iter_, const double& mu_, 
		    const double &f, const hiopVector& c_, const hiopVector& d_, 
		    const hiopVector& gradf_,  const hiopMatrixDense& Jac_c_,  const hiopMatrixDense& Jac_d_) 
  {
    mu=mu_; c_nlp=&c_; d_nlp=&d_; Jac_c_nlp=&Jac_c_; Jac_d_nlp=&Jac_d_; iter=&iter_;
    _grad_x_logbar->copyFrom(gradf_);
    _grad_d_logbar->setToZero(); double aux=-mu * iter->evalLogBarrier();
    f_logbar = f + aux;
    if(kappa_d>0.) {
      iter->addLinearDampingTermToGrad_x(mu,kappa_d,1.0,*_grad_x_logbar);
      iter->addLinearDampingTermToGrad_d(mu,kappa_d,1.0,*_grad_d_logbar);

      f_logbar += iter->linearDampingTerm(mu,kappa_d);
    }
  }
  inline void 
  updateWithNlpInfo_trial_funcOnly(const hiopIterate& iter_, 
				   const double &f, const hiopVector& c_, const hiopVector& d_)
  {
    c_nlp_trial=&c_; d_nlp_trial=&d_; iter_trial=&iter_;
    f_logbar_trial = f - mu * iter_trial->evalLogBarrier();
    if(kappa_d>0.) f_logbar_trial += iter_trial->linearDampingTerm(mu,kappa_d);
  }
  /* gradx += beta*grad_x_logBar*/
  inline void addLogBarTermsToGrad_x(const double& beta, hiopVector& gradx) const
  {
    if(kappa_d>0.) iter->addLinearDampingTermToGrad_x(mu,kappa_d,beta,gradx);
  }
  /* gradd += beta*grad_d_logBar*/
  inline void addLogBarTermsToGrad_d(const double& beta, hiopVector& gradd) const
  {
    if(kappa_d>0.) iter->addLinearDampingTermToGrad_d(mu,kappa_d,beta,gradd);
  }
  /* grad_log^T * [ dx ] =  grad_f^T * dx + grad_x_dampingTerm^T * dx + grad_d_dampingTerm^T *ds 
                  [ dd ]   
  */
  inline double directionalDerivative(const hiopIterate& dir) 
  {
    double tr = dir.get_x()->dotProductWith(*_grad_x_logbar);
    tr       += dir.get_d()->dotProductWith(*_grad_d_logbar);
    return tr;
  }

protected:
  hiopNlpDenseConstraints* nlp;
private:
  hiopLogBarProblem() {};
  hiopLogBarProblem(const hiopLogBarProblem&) {};
  hiopLogBarProblem& operator=(const hiopLogBarProblem&) {return *this;};
};

#endif
