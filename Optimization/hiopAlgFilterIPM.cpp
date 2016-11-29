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

  _Hess    = new hiopHessianInvLowRank(nlp,10);

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  //default values for the parameters
  mu0=_mu=0.1; 
  kappa_mu=0.2; //linear decrease factor
  theta_mu=1.5; //exponent for higher than linear decrease of mu
  tau_min=0.99; //min value for the fraction-to-the-boundary
  eps_tol=1e-8; //absolute error for the nlp
  kappa_eps=10; //relative (to mu) error for the log barrier
  kappa1=kappa2=1e-2; //projection params for the starting point (default 1e-2)
  p_smax=100; //theshold for the magnitude of the multipliers

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
  if(_Jac_d_trial)   delete _Jac_d;

  if(resid_trial)    delete resid_trial;
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

  nlp->log->write("Initial point:", it_ini, hovIteration);

  return true;
}

int hiopAlgFilterIPM::run()
{
  defaultStartingPoint(*it_curr);
  _mu=mu0;
  //update problem information and residuals
  updateLogBarrierProblem(*it_curr, _mu, _f_nlp, _f_log, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d,_mu);
  nlp->log->write("First residual:", *resid, hovIteration);

  iter_num=0;

  theta_max=1e+4*fmax(1.0,resid->getNlpInfeasInfNorm());
  theta_min=1e-4*fmax(1.0,resid->getNlpInfeasInfNorm());
  
  hiopKKTLinSysLowRank* kkt=new hiopKKTLinSysLowRank(nlp);

  _alpha_primal = _alpha_dual = 0;
  
  bool bStopAlg=false; int nAlgStatus=0; bool bret=true;
  while(!bStopAlg) {
    bret = evalNlpAndLogErrors(*it_curr, *resid, _mu, 
			       _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp, 
			       _err_log_optim, _err_log_feas, _err_log_complem, _err_log); 
    assert(bret);

    outputIteration();
 
    if(_err_nlp<=eps_tol) { bStopAlg=1; nAlgStatus=1; break; }
    if(_err_log<=kappa_eps * _mu) {
      //update mu and tau (fraction-to-boundary)
      bret = updateLogBarrierParameters(*it_curr, _mu, _tau, _mu, _tau);

      //update problem information and residuals
      updateLogBarrierProblem(*it_curr, _mu, _f_nlp, _f_log, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
      resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d,_mu);

      filter.reinitialize(theta_max);
      //recheck residuals for at the first iteration in case the starting pt is  very good
      if(iter_num==0) {
	continue; 
      }
    }
    // --- search direction calculation ---
    //first update kkt system
    kkt->update(it_curr,_grad_f,_Jac_c,_Jac_d, _Hess);
    bret = kkt->computeDirections(resid,dir); assert(bret==true);

    /***************************************************************
     * backtracking line search
     */
    //max steps
    bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual); assert(bret);
    //    alphaprimal=alphadual=1e-4;
    double theta = thetaLogBarrier(*it_curr, *resid, _mu);
    //printf("iter=%4d theta=%10.5e phi=%10.5e \n", num_iter,theta,_f);
    //printf("iter=%4d fractionToTheBdry steps (%g,%g)\n", num_iter, alphaprimal, alphadual);
    //Armijo line search
    bool bSearchDone=false;
    while(!bSearchDone) {

      bret = it_trial->updatePrimals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
  
     

      //it_curr->print();
      
      //update problem information and residuals
      updateLogBarrierProblem(*it_trial, _mu, _f_nlp_trial, _f_log_trial, *_c_trial, *_d_trial, *_grad_f_trial, *_Jac_c_trial, *_Jac_d_trial);
      resid_trial->update(*it_trial,_f_nlp_trial, *_c_trial, *_d_trial, *_grad_f_trial, *_Jac_c_trial, *_Jac_d_trial, _mu);
      
      //theta and phi
      double theta_plus=thetaLogBarrier(*it_trial,*resid_trial, _mu);
      //printf("iter=%4d theta=%10.5e phi=%10.5e (trial)\n", num_iter,theta_plus,_f_trial);

      if(_f_nlp_trial>_f_nlp) _alpha_primal *= 0.9;
      else break;
    } 
    bret = it_trial->updateDualsIneq(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
    bret = it_trial->updateDualsEq(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
    it_curr->copyFrom(*it_trial);
    //updateLogBarrierProblem(*it_curr, mu, _f, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, _mu);
    iter_num++;
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



bool hiopAlgFilterIPM::
updateLogBarrierProblem(hiopIterate& iter, double mu, 
			double &f, double& f_log, hiopVector& c_, hiopVector& d_, 
			hiopVector& grad_,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d)
{
  bool new_x=true, bret; 
  const hiopVectorPar& it_x = dynamic_cast<const hiopVectorPar&>(*iter.get_x());
  hiopVectorPar 
    &c=dynamic_cast<hiopVectorPar&>(c_), 
    &d=dynamic_cast<hiopVectorPar&>(d_), 
    &grad=dynamic_cast<hiopVectorPar&>(grad_);
  const double* x = it_x.local_data_const();//local_data_const();
  //f(x)
  bret = nlp->eval_f(x, new_x, f); assert(bret);
  new_x=false; //same x for the rest
  bret = nlp->eval_grad_f(x, new_x, grad.local_data());  assert(bret);
  bret = nlp->eval_c     (x, new_x, c.local_data());     assert(bret);
  bret = nlp->eval_d     (x, new_x, d.local_data());     assert(bret);
  bret = nlp->eval_Jac_c (x, new_x, Jac_c.local_data()); assert(bret);
  bret = nlp->eval_Jac_d (x, new_x, Jac_d.local_data()); assert(bret);

  //add the log barrier term
  f_log = 0.0 - mu * iter.evalLogBarrier();
  f_log += f;
  return true;
}


void hiopAlgFilterIPM::outputIteration()
{
  if(iter_num/10*10==iter_num) 
    printf("iter    objective     inf_pr     inf_du    lg(mu)  ||d||    lg(rg)   alpha_du    alpha_pr  ls\n");
  printf("%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e     -     %7.3e   %7.3e  -\n",
	 iter_num, _f_nlp, _err_nlp_feas, _err_nlp_optim, log10(_mu), 0.0, _alpha_dual, _alpha_primal); 

}
