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

  _f = 0; 
  _c = nlp->alloc_dual_eq_vec(); 
  _d = nlp->alloc_dual_ineq_vec();

  _grad_f  = nlp->alloc_primal_vec();
  _Jac_c   = nlp->alloc_Jac_c();
  _Jac_d   = nlp->alloc_Jac_d();

  _f_trial=0;
  _c_trial = nlp->alloc_dual_eq_vec(); 
  _d_trial = nlp->alloc_dual_ineq_vec();

  _grad_f_trial  = nlp->alloc_primal_vec();
  _Jac_c_trial   = nlp->alloc_Jac_c();
  _Jac_d_trial   = nlp->alloc_Jac_d();

  _Hess    = new hiopHessianInvLowRank(nlp,10);

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  //default values for the parameters
  mu0=mu=0.01; 
  kappa_mu=0.2; //linear decrease factor
  theta_mu=1.5; //exponent for higher than linear decrease of mu
  tau_min=0.99; //min value for the fraction-to-the-boundary
  eps_tol=1e-8; //absolute error for the nlp
  kappa_eps=10; //relative (to mu) error for the log barrier
  kappa1=kappa2=2e-1; //projection params for the starting point (default 1e-2)
  smax=100; //theshold for the magnitude of the multipliers

  tau=fmax(tau_min,1.0-mu);
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
  it_ini.projectPrimalsIntoBounds(kappa1, kappa2);
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
  mu=mu0;
  //update problem information and residuals
  updateLogBarrierProblem(*it_curr, mu, _f, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  resid->update(*it_curr,_f, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d,mu);

  theta_max=1e+4*fmax(1.0,resid->getNlpInfeasInfNorm());
  theta_min=1e-4*fmax(1.0,resid->getNlpInfeasInfNorm());
  
  hiopKKTLinSysLowRank* kkt=new hiopKKTLinSysLowRank(nlp);

  double err_nlp, err_log; double alphaprimal, alphadual;
  int num_iter=0;
  bool bStopAlg=false; int nAlgStatus=0; bool bret=true;
  while(!bStopAlg) {

    //overall and log-barrier problems convergence checks
    bret = nlpAndLogBarrierErrors(*it_curr,*resid, mu, err_nlp, err_log); assert(bret);
    printf("iter=%4d mu=%10.5e nlpErr=%10.5f logErr=%10.5f\n", num_iter,mu,err_nlp, err_log);
    if(err_nlp<=eps_tol) { bStopAlg=1; nAlgStatus=1; break; }
    if(err_log<=kappa_eps*mu) {
      //update mu and tau (fraction-to-boundary)
      bret = updateLogBarrierParameters(*it_curr, mu,tau, mu,tau);

      //update problem information and residuals
      updateLogBarrierProblem(*it_curr, mu, _f, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
      resid->update(*it_curr,_f, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d,mu);

      filter.reinitialize(theta_max);
      //recheck residuals for at the first iteration in case the starting pt is  very good
      if(num_iter==0) {
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
    bret = it_curr->fractionToTheBdry(*dir, tau, alphaprimal, alphadual); assert(bret);
    //    alphaprimal=alphadual=1e-4;
    double theta = thetaLogBarrier(*it_curr, *resid, mu);
    printf("iter=%4d theta=%10.5e phi=%10.5e \n", num_iter,theta,_f);
    printf("iter=%4d fractionToTheBdry steps (%g,%g)\n", num_iter, alphaprimal, alphadual);
    //Armijo line search
    bool bSearchDone=false;
    while(!bSearchDone) {
      printf("iter=%4d steps (%g,%g)\n", num_iter, alphaprimal, alphadual);
      bret = it_trial->updatePrimals(*it_curr, *dir, alphaprimal, alphadual); assert(bret);
  
     

      //it_curr->print();
      
      //update problem information and residuals
      updateLogBarrierProblem(*it_trial, mu, _f_trial, *_c_trial, *_d_trial, *_grad_f_trial, *_Jac_c_trial, *_Jac_d_trial);
      resid_trial->update(*it_trial,_f_trial, *_c_trial, *_d_trial, *_grad_f_trial, *_Jac_c_trial, *_Jac_d_trial, mu);
      
      //theta and phi
      double theta_plus=thetaLogBarrier(*it_trial,*resid_trial, mu);
      printf("iter=%4d theta=%10.5e phi=%10.5e (trial)\n", num_iter,theta_plus,_f_trial);

      if(_f_trial>_f) alphaprimal *= 0.9;
      else break;
    } 
    bret = it_trial->updateDualsIneq(*it_curr, *dir, alphaprimal, alphadual); assert(bret);
    bret = it_trial->updateDualsEq(*it_curr, *dir, alphaprimal, alphadual); assert(bret);
    it_curr->copyFrom(*it_trial);
    updateLogBarrierProblem(*it_curr, mu, _f, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    resid->update(*it_curr,_f, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d,mu);
    num_iter++;
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


double hiopAlgFilterIPM::nlpError(const hiopIterate& it, const hiopResidual& resid)
{
  long long n=nlp->n_complem(), m=nlp->m();
  //the one norms
  double nrmDualBou=it.normOneOfBoundDuals();
  double nrmDualEqu=it.normOneOfEqualityDuals();
  //scaling factors
  double sd = fmax(smax,(nrmDualBou+nrmDualEqu)/(n+m)) / smax;
  double sc = n==0?0:fmax(smax,nrmDualBou/n) / smax;
  //actual nlp errors
  double optim, feas, complem;
  resid.getNlpErrors(optim, feas, complem);

  //finally, the scaled nlp error
  return fmax(optim/sd, fmax(feas, complem/sc));
}


//double hiopAlgFilterIPM::

double hiopAlgFilterIPM::barrierError(const hiopIterate& it, const hiopResidual& resid, const double& mu)
{
  long long n=nlp->n_complem(), m=nlp->m();
  //the one norms
  double nrmDualBou=it.normOneOfBoundDuals();
  double nrmDualEqu=it.normOneOfEqualityDuals();
  //scaling factors
  double sd = fmax(smax,(nrmDualBou+nrmDualEqu)/(n+m)) / smax;
  double sc = n==0?0:fmax(smax,nrmDualBou/n) / smax;
  //actual nlp errors
  double optim, feas, complem;
  resid.getBarrierErrors(optim, feas, complem);

  //finally, the scaled barrier error
  return fmax(optim/sd, fmax(feas, complem/sc));
}


bool hiopAlgFilterIPM::nlpAndLogBarrierErrors(const hiopIterate& it, const hiopResidual& resid, const double& mu, 
					      double& err_nlp, double& err_log)
{
  long long n=nlp->n_complem(), m=nlp->m();
  //the one norms
  double nrmDualBou=it.normOneOfBoundDuals();
  double nrmDualEqu=it.normOneOfEqualityDuals();
  //scaling factors
  double sd = fmax(smax,(nrmDualBou+nrmDualEqu)/(n+m)) / smax;
  double sc = n==0?1:fmax(smax,nrmDualBou/n) / smax;
  //actual nlp errors
  double optim_nlp, feas_nlp, complem_nlp;
  resid.getNlpErrors(optim_nlp, feas_nlp, complem_nlp);
  //actual log-barrier errors
  double optim_log, feas_log, complem_log;
  resid.getBarrierErrors(optim_log, feas_log, complem_log);
  //finally, the scaled  errors
  err_nlp = fmax(optim_nlp/sd, fmax(feas_nlp, complem_nlp/sc));
  err_log = fmax(optim_log/sd, fmax(feas_log, complem_log/sc));

  printf("errors barrier: [%12.6e] [%12.6e] [%12.6e] |  [%12.6e] [%12.6e] [%12.6e]\n",
	 optim_log, feas_log, complem_log, optim_log/sd, feas_log, complem_log/sc);  
  printf("errors nlp:     [%12.6e] [%12.6e] [%12.6e] |  [%12.6e] [%12.6e] [%12.6e]\n",
	 optim_nlp, feas_nlp, complem_nlp, optim_nlp/sd, feas_nlp, complem_nlp/sc);  

  return true;
}


bool hiopAlgFilterIPM::
updateLogBarrierProblem(hiopIterate& iter, double mu, 
			double &f, hiopVector& c_, hiopVector& d_, 
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
  f-=mu * iter.evalLogBarrier();
  it_x.print("x   iterate:");
  iter.get_sxl()->print("sxl iterate");
  return true;
}
