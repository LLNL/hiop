#include "hiopAlgFilterIPM.hpp"
#include "hiopKKTLinSys.hpp"



#include <cmath>

#include <cassert>

hiopAlgFilterIPM::hiopAlgFilterIPM(hiopNlpDenseConstraints* nlp_)
  : nlp(nlp_)
{
  it_curr = new hiopIterate(nlp);
  it_trial= it_curr->alloc_clone();
  dir     = it_curr->alloc_clone();

  logbar = new hiopLogBarProblem(nlp);

  _f_nlp = _f_log = 0; 
  _c = nlp->alloc_dual_eq_vec(); 
  _d = nlp->alloc_dual_ineq_vec();

  _grad_f  = nlp->alloc_primal_vec();
  _Jac_c   = nlp->alloc_Jac_c();
  _Jac_d   = nlp->alloc_Jac_d();

  _f_nlp_trial = _f_log_trial = 0;
  _c_trial = nlp->alloc_dual_eq_vec(); 
  _d_trial = nlp->alloc_dual_ineq_vec();

  _grad_f_trial  = nlp->alloc_primal_vec();
  _Jac_c_trial   = nlp->alloc_Jac_c();
  _Jac_d_trial   = nlp->alloc_Jac_d();

  _Hess    = new hiopHessianLowRank(nlp,10);

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  //default values for the parameters
  mu0=_mu=0.1; 
  kappa_mu=0.2;       //linear decrease factor
  theta_mu=1.5;       //exponent for higher than linear decrease of mu
  tau_min=0.99;       //min value for the fraction-to-the-boundary
  eps_tol=1e-8;       //absolute error for the nlp
  kappa_eps=10;       //relative (to mu) error for the log barrier
  kappa1=kappa2=1e-2; //projection params for the starting point (default 1e-2)
  p_smax=100;         //threshold for the magnitude of the multipliers
  gamma_theta=1e-5;   //sufficient progress parameters for the feasibility violation
  gamma_phi=1e-5;     //and log barrier objective
  s_theta=1.1;        //parameters in the switch condition of 
  s_phi=2.3;          // the linearsearch (equation 19) in
  delta=1.;           // the WachterBiegler paper
  eta_phi=1e-4;       // parameter in the Armijo rule
  kappa_Sigma = 1e10; //parameter in resetting the duals to guarantee closedness of the primal-dual logbar Hessian to the primal logbar Hessian
  _tau=fmax(tau_min,1.0-_mu);
  theta_max = 1e7; //temporary - will be updated after ini pt is computed
  theta_min = 1e7; //temporary - will be updated after ini pt is computed
}

hiopAlgFilterIPM::~hiopAlgFilterIPM()
{
  if(it_curr)  delete it_curr;
  if(it_trial) delete it_trial;
  if(dir)      delete dir;

  if(_c)       delete _c;
  if(_d)       delete _d;
  if(_grad_f)  delete _grad_f;
  if(_Jac_c)   delete _Jac_c;
  if(_Jac_d)   delete _Jac_d;
  if(_Hess)    delete _Hess;

  if(resid)    delete resid;

  if(_c_trial)       delete _c_trial;
  if(_d_trial)       delete _d_trial;
  if(_grad_f_trial)  delete _grad_f_trial;
  if(_Jac_c_trial)   delete _Jac_c_trial;
  if(_Jac_d_trial)   delete _Jac_d_trial;

  if(resid_trial)    delete resid_trial;

  if(logbar) delete logbar;
}

int hiopAlgFilterIPM::defaultStartingPoint(hiopIterate& it_ini)
{
  if(!nlp->get_starting_point(*it_ini.get_x())) {
    printf("error: in getting the user provided starting point");
    assert(false); return false;
  }
  it_ini.projectPrimalsXIntoBounds(kappa1, kappa2);

  if(!nlp->eval_d(*it_ini.get_x(), true, *it_ini.get_d())) {
    printf("error: in user provided constraint function");
    assert(false); return false;
  }
  it_ini.projectPrimalsDIntoBounds(kappa1, kappa2);

  it_ini.determineSlacks();

  it_ini.setBoundsDualsToConstant(1.);

  //-- //! lsq for yd and yc
  //for now set them to zero
  it_ini.setEqualityDualsToConstant(0.);

  return true;
}

int hiopAlgFilterIPM::run()
{
  defaultStartingPoint(*it_curr);
  _mu=mu0;
  //update problem information 
  this->evalNlp(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  //update log bar
  logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  nlp->log->printf(hovScalars, "log bar obj: %g", logbar->f_logbar);
  //recompute the residuals
  resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);

  nlp->log->write("First residual-------------", *resid, hovIteration);

  iter_num=0;

  theta_max=1e+4*fmax(1.0,resid->getInfeasInfNorm());
  theta_min=1e-4*fmax(1.0,resid->getInfeasInfNorm());
  
  hiopKKTLinSysLowRank* kkt=new hiopKKTLinSysLowRank(nlp);

  _alpha_primal = _alpha_dual = 0;
  
  bool bStopAlg=false; int nAlgStatus=0; bool bret=true;
  while(!bStopAlg) {

    bret = evalNlpAndLogErrors(*it_curr, *resid, _mu, 
			       _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp, 
			       _err_log_optim, _err_log_feas, _err_log_complem, _err_log); assert(bret);

    nlp->log->printf(hovScalars, "  Nlp    errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n",
		     _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
    nlp->log->printf(hovScalars, "  LogBar errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n",
		     _err_log_feas, _err_log_optim, _err_log_complem, _err_log);

    outputIteration();
 
    if(_err_nlp<=eps_tol) { bStopAlg=1; nAlgStatus=1; break; }
    if(_err_log<=kappa_eps * _mu) {
      //update mu and tau (fraction-to-boundary)
      bret = updateLogBarrierParameters(*it_curr, _mu, _tau, _mu, _tau);
      nlp->log->printf(hovScalars, "Iter[%d] barrier params reduced: mu=%g tau=%g\n", iter_num, _mu, _tau);

      //update only logbar problem  and residual (the NLP didn't change)
      //this->evalNlp(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
      logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
      resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar); //! should perform only a partial update since NLP didn't change
      bret = evalNlpAndLogErrors(*it_curr, *resid, _mu, 
				 _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp, 
				 _err_log_optim, _err_log_feas, _err_log_complem, _err_log); assert(bret);
      nlp->log->printf(hovScalars, "  Nlp    errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n",
		       _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
      nlp->log->printf(hovScalars, "  LogBar errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n",
		       _err_log_feas, _err_log_optim, _err_log_complem, _err_log);    
      
      filter.reinitialize(theta_max);
      //recheck residuals for at the first iteration in case the starting pt is  very good
      if(iter_num==0) {
	continue; 
      }
    }
    nlp->log->printf(hovScalars, "Iter[%d] logbarObj=%20.14e (mu=%12.5e)\n", iter_num, logbar->f_logbar,_mu);
    // --- search direction calculation ---
    //first update the Hessian and kkt system
    _Hess->update(*it_curr,*_grad_f,*_Jac_c,*_Jac_d);
    kkt->update(it_curr,_grad_f,_Jac_c,_Jac_d, _Hess);
    bret = kkt->computeDirections(resid,dir); assert(bret==true);
    nlp->log->printf(hovIteration, "Iter[%d] full search direction -------------\n"); nlp->log->write("", *dir, hovIteration);

    /***************************************************************
     * backtracking line search
     ****************************************************************/
    //max step
    bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual); assert(bret);
    double theta = resid->getInfeasInfNorm(); //at it_curr
    double theta_trial;

    bool grad_phi_dx_computed=false; double grad_phi_dx;
    
    while(true) {
      //check the step against the minimum step size
      if(_alpha_primal<1e-16) {
	nlp->log->write("Panic: minimum step size reached",hovSummary);
	assert(false);
      }

      bret = it_trial->takeStep_primals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
      //evaluate the problem at the trial iterate (functions only)
      this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);
      logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

      //compute infeasibility theta at trial point.
      theta_trial = resid->computeNlpInfeasInfNorm(*it_trial, *_c_trial, *_d_trial);

      nlp->log->printf(hovLinesearch, "  trial point: alphaPrimal=%14.8e barier:(%15.9e)>%15.9e theta:(%15.9e)>%15.9e\n",
		       _alpha_primal, logbar->f_logbar, logbar->f_logbar_trial, theta, theta_trial);

      bool switchingCondTrue=false;
      //let's do the cheap, "sufficient progress" test first, before more involved/expensive tests. 
      // This simple test is good enough when iterate is far away from solution
      if(theta_trial>=theta_min) {
	//check the filter and the sufficient decrease condition (18)
	if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
	  if(theta_trial<=(1-gamma_theta)*theta || logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {
	    //trial good to go
	    break;
	  } else {
	    //there is no sufficient progress 
	    _alpha_primal *= 0.5;
	    continue;
	  }
	} else {
	  //it is in the filter 
	  _alpha_primal *= 0.5;
	  continue;
	}  
      } else {
	if(theta_trial<=theta_min) { 
	  // if(theta_trial<theta_min  then check the switching condition and Armijo rule
	  // first compute grad_phi^T d_x if it hasn't already been computed
	  if(!grad_phi_dx_computed) { grad_phi_dx = logbar->directionalDerivative(*dir); grad_phi_dx_computed=true; }
	  //this is the actual switching condition
	  if(grad_phi_dx<0 && _alpha_primal*pow(grad_phi_dx,s_phi)>delta*pow(theta,s_theta)) {
	    switchingCondTrue=true;
	    if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi*_alpha_primal*grad_phi_dx)
	      break; //iterate good to go since it satisfies Armijo
	    else {  //Armijo is not satisfied
	      _alpha_primal *= 0.5; //reduce step and try again
	      continue;
	    }
	  } // else: switching condition does not hold  and switchingCondTrue remains to false
	}
      }

      if(!switchingCondTrue) {
	//ok to go with  "sufficient progress" condition even when close to solution
	//check the filter and the sufficient decrease condition (18)
	if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
	  //if(theta_trial<=(1-gamma_theta)*theta || logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {
          if(logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {
	    //trial good to go
	    break;
	  } else {
	    //there is no sufficient progress 
	    _alpha_primal *= 0.5;
	    continue;
	  }
	} else {
	  //it is in the filter 
	  _alpha_primal *= 0.5;
	  continue;
	} 
      } 

      assert(false); //shouldn't get here
    }
    
    
    //post line-search stuff: such as update filter
    // to be done
    
    nlp->log->printf(hovScalars, "Iter[%d] -> accepted step primal=[%17.11e] dual=[%17.11e]\n", iter_num, _alpha_primal, _alpha_dual);
    iter_num++;
    //update and reset the duals
    bret = it_trial->takeStep_duals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
    bret = it_trial->adjustDuals_primalLogHessian(_mu,kappa_Sigma); assert(bret);
    
    //update current iterate (do a fast swap of the pointers)
    hiopIterate* pit=it_curr; it_curr=it_trial; it_trial=pit;
    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate -------------", iter_num); nlp->log->write("", *it_curr, hovIteration);

    this->evalNlp_derivOnly(*it_curr, *_grad_f, *_Jac_c, *_Jac_d);
    //reuse function values
    _f_nlp=_f_nlp_trial; hiopVector* pvec=_c_trial; _c_trial=_c; _c=pvec; pvec=_d_trial; _d_trial=_d; _d=pvec;
   
    logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    //update residual
    resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);
    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num); nlp->log->write("", *resid, hovIteration);

    if(iter_num>=50) break;
  }

  delete kkt;

  return true;
}

bool hiopAlgFilterIPM::
updateLogBarrierParameters(const hiopIterate& it, const double& mu_curr, const double& tau_curr,
			   double& mu_new, double& tau_new)
{
  mu_new  = fmax(eps_tol/10, fmin(kappa_mu*mu_curr, pow(mu_curr,theta_mu)));
  tau_new = fmax(tau_min,1.0-mu_new);
  return true;
}
double hiopAlgFilterIPM::thetaLogBarrier(const hiopIterate& it, const hiopResidual& resid, const double& mu)
{
  //actual nlp errors
  double optim, feas, complem;
  resid.getNlpErrors(optim, feas, complem);
  return feas;
}


bool hiopAlgFilterIPM::
evalNlpAndLogErrors(const hiopIterate& it, const hiopResidual& resid, const double& mu,
		    double& nlpoptim, double& nlpfeas, double& nlpcomplem, double& nlpoverall,
		    double& logoptim, double& logfeas, double& logcomplem, double& logoverall)
{
  long long n=nlp->n_complem(), m=nlp->m();
  //the one norms
  double nrmDualBou=it.normOneOfBoundDuals();
  double nrmDualEqu=it.normOneOfEqualityDuals();
  //scaling factors
  double sd = fmax(p_smax,(nrmDualBou+nrmDualEqu)/(n+m)) / p_smax;
  double sc = n==0?0:fmax(p_smax,nrmDualBou/n) / p_smax;
  //actual nlp errors 
  resid.getNlpErrors(nlpoptim, nlpfeas, nlpcomplem);

  //finally, the scaled nlp error
  nlpoverall = fmax(nlpoptim/sd, fmax(nlpfeas, nlpcomplem/sc));

  //actual log errors
  resid.getBarrierErrors(logoptim, logfeas, logcomplem);

  //finally, the scaled barrier error
  logoverall = fmax(logoptim/sd, fmax(logfeas, logcomplem/sc));

  return true;
}



bool hiopAlgFilterIPM::evalNlp(hiopIterate& iter, 			       
			       double &f, hiopVector& c_, hiopVector& d_, 
			       hiopVector& gradf_,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d)
{
  bool new_x=true, bret; 
  const hiopVectorPar& it_x = dynamic_cast<const hiopVectorPar&>(*iter.get_x());
  hiopVectorPar 
    &c=dynamic_cast<hiopVectorPar&>(c_), 
    &d=dynamic_cast<hiopVectorPar&>(d_), 
    &gradf=dynamic_cast<hiopVectorPar&>(gradf_);
  const double* x = it_x.local_data_const();//local_data_const();
  //f(x)
  bret = nlp->eval_f(x, new_x, f); assert(bret);
  new_x= false; //same x for the rest
  bret = nlp->eval_grad_f(x, new_x, gradf.local_data());  assert(bret);

  bret = nlp->eval_c     (x, new_x, c.local_data());     assert(bret);
  bret = nlp->eval_d     (x, new_x, d.local_data());     assert(bret);
  bret = nlp->eval_Jac_c (x, new_x, Jac_c.local_data()); assert(bret);
  bret = nlp->eval_Jac_d (x, new_x, Jac_d.local_data()); assert(bret);

  return bret;
}

bool hiopAlgFilterIPM::evalNlp_funcOnly(hiopIterate& iter,
					double& f, hiopVector& c_, hiopVector& d_)
{
  bool new_x=true, bret; 
  const hiopVectorPar& it_x = dynamic_cast<const hiopVectorPar&>(*iter.get_x());
  hiopVectorPar 
    &c=dynamic_cast<hiopVectorPar&>(c_), 
    &d=dynamic_cast<hiopVectorPar&>(d_);
  const double* x = it_x.local_data_const();
  bret = nlp->eval_f(x, new_x, f); assert(bret);
  new_x= false; //same x for the rest
  bret = nlp->eval_c(x, new_x, c.local_data());     assert(bret);
  bret = nlp->eval_d(x, new_x, d.local_data());     assert(bret);
  return bret;
}
bool hiopAlgFilterIPM::evalNlp_derivOnly(hiopIterate& iter,
					 hiopVector& gradf_,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d)
{
  bool new_x=false; //functions were previously evaluated in the line search
  bool bret;
  const hiopVectorPar& it_x = dynamic_cast<const hiopVectorPar&>(*iter.get_x());
  hiopVectorPar & gradf=dynamic_cast<hiopVectorPar&>(gradf_);
  const double* x = it_x.local_data_const();
  bret = nlp->eval_grad_f(x, new_x, gradf.local_data()); assert(bret);
  bret = nlp->eval_Jac_c (x, new_x, Jac_c.local_data()); assert(bret);
  bret = nlp->eval_Jac_d (x, new_x, Jac_d.local_data()); assert(bret);
  return bret;
}

void hiopAlgFilterIPM::outputIteration()
{
  if(iter_num/10*10==iter_num) 
    printf("iter    objective     inf_pr     inf_du   lg(mu)  alpha_du   alpha_pr\n");
  printf("%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e\n",
	 iter_num, _f_nlp, _err_nlp_feas, _err_nlp_optim, log10(_mu), _alpha_dual, _alpha_primal); 

}
