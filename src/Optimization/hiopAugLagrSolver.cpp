#include "hiopAugLagrSolver.hpp"
#include "hiopResidualAugLagr.hpp"

#include <algorithm>    // std::max
#include <math.h>       /* pow */


namespace hiop
{

hiopAugLagrSolver::hiopAugLagrSolver(NLP_CLASS_IN* nlp_in_) 
  : nlp(new hiopAugLagrNlpAdapter(nlp_in_)),
    subproblemSolver(nlp),
    n_vars(-1),m_cons(-1),
    _it_curr(nullptr),
    _lam_curr(nullptr),
    _rho_curr(-1),
    residual(nullptr),
    _zL(nullptr), _zU(nullptr),
    _solverStatus(NlpSolve_IncompleteInit),
    _iter_num(-1), _n_accep_iters(-1),
    _f_nlp(-1.),
    _err_feas0(-1.), _err_optim0(-1.),
    _err_feas(-1.), _err_optim(-1.),
    _nrm_dlam(0.),
    _LAMBDA0(-1.),_WARM_INIT_LAMBDA(false),_RHO0(-1.),
    _EPS_TOL(-1),_EPS_RTOL(-1),_EPS_TOL_ACCEP(-1),
    _MAX_N_IT(-1), _ACCEP_N_IT(-1),
    _alpha(-1.),
    _eps_tol_feas0(-1.), _eps_tol_feas(-1.),
    _eps_tol_optim0(-1.), _eps_tol_optim(-1.)
{
  // get size of the problem and the penalty term
  long long dum1;
  nlp->get_prob_sizes(n_vars, dum1);
  nlp->get_penalty_size(m_cons);
  
  _it_curr  = new hiopVectorPar(n_vars);
  _lam_curr = new hiopVectorPar(m_cons);
  residual  = new hiopResidualAugLagr(nlp, n_vars, m_cons);
  _zL  = new hiopVectorPar(n_vars);
  _zU  = new hiopVectorPar(n_vars);

  reloadOptions();
  reInitializeNlpObjects();
  resetSolverStatus();
}

hiopAugLagrSolver::~hiopAugLagrSolver() 
{
    if(nlp)       delete nlp;
    if(_it_curr)  delete _it_curr;
    if(_lam_curr) delete _lam_curr;
    if(residual)  delete residual;
    if(_zL)       delete _zL;
    if(_zU)       delete _zU;
}

void hiopAugLagrSolver::reloadOptions()
{
  // initial value of the multipliers and the penalty
  _WARM_INIT_LAMBDA = nlp->options->GetInteger("warm_start");
  _LAMBDA0 = nlp->options->GetNumeric("lambda0");
  _RHO0 = nlp->options->GetNumeric("rho0");

  //user options
  _EPS_TOL  = nlp->options->GetNumeric("tolerance");          ///< abs tolerance for the NLP error (same for feas and optim)
  _EPS_RTOL = nlp->options->GetNumeric("rel_tolerance");      ///< rel tolerance for the NLP error (same for feas and optim)
  _EPS_TOL_ACCEP = nlp->options->GetNumeric("acceptable_tolerance"); ///< acceptable tolerance (required at _ACCEP_N_IT iterations)
  _MAX_N_IT   = nlp->options->GetInteger("max_iter");                ///< maximum number of iterations
  _ACCEP_N_IT = nlp->options->GetInteger("acceptable_iterations");   ///< acceptable number of iterations

  // internal algorithm parameters
  _alpha = 1./_RHO0; ///< positive constants
  _eps_tol_feas0  = 1e-1 * _alpha; ///< required feasibility of the subproblem
  _eps_tol_optim0 = 1e-2 * pow(_alpha, 0.1); ///< required optimality tolerance of the subproblem
  _eps_tol_feas0  = _eps_tol_feas0;  //1e-3;
  _eps_tol_optim0 = _eps_tol_optim0; //1e-3;
  
  _eps_tol_feas  = _eps_tol_feas0;
  _eps_tol_optim = _eps_tol_optim0;
}

void hiopAugLagrSolver::resetSolverStatus()
{
  _iter_num = 0;
  _n_accep_iters = 0;
  _solverStatus = NlpSolve_IncompleteInit;
  _err_feas0 = -1.;
  _err_optim0 = -1;
  _nrm_dlam = 0.;
}

void hiopAugLagrSolver::reInitializeNlpObjects()
{
    //TODO
}

/**
 */
hiopSolveStatus hiopAugLagrSolver::run()
{

  nlp->log->printf(hovSummary, "==================\nHiop AugLagr SOLVER\n==================\n");

  reloadOptions();
  reInitializeNlpObjects();
  resetSolverStatus();

  subproblemSolver.initialize();

  _solverStatus = NlpSolve_SolveNotCalled;
  
  nlp->runStats.initialize();
  nlp->runStats.tmOptimizTotal.start();
  nlp->runStats.tmStartingPoint.start();

  //initialize curr_iter by calling TNLP starting point + do something about slacks
  // and set starting point at the Adapter class for the first major AL iteration
  nlp->get_user_starting_point(n_vars, _it_curr->local_data(), _WARM_INIT_LAMBDA , _lam_curr->local_data());
  nlp->set_starting_point(n_vars, _it_curr->local_data_const());
  
  std::string name = "iter" + std::to_string(_iter_num) + ".txt";
  FILE *f=fopen(name.c_str(),"w");
  _it_curr->print(f);
  fclose(f);

  //set initial guess of the multipliers and the penalty parameter
  if (!_WARM_INIT_LAMBDA )
  {
    _lam_curr->setToConstant(_LAMBDA0);
  }
  nlp->set_lambda(_lam_curr);
  
  _rho_curr = _RHO0;
  nlp->set_rho(_rho_curr);
  
  nlp->runStats.tmStartingPoint.stop();
  
  //initial evaluation of the problem
  nlp->runStats.nIter=_iter_num;

  //evaluate the problem
  evalNlp(_it_curr, _f_nlp);

  //evalate initial primal/dual infeasibility
  _zL->setToConstant(0.);
  _zU->setToConstant(0.); //how to initialize zL,zU? we need to solve the subproblem to get their values
  evalNlpErrors(_it_curr, _zL, _zU, residual, _err_feas, _err_optim);
  nlp->log->write("First residual-------------", *residual, hovIteration);
   
  name = "resid" + std::to_string(_iter_num) + ".txt";
  FILE *f43=fopen(name.c_str(),"w");
  residual->print(f43);
  fclose(f43);

  //check termination conditions   
  bool notConverged = true;
  if(checkTermination(_err_feas, _err_optim, _iter_num, _solverStatus)) {
      notConverged = false;
    }
  outputIteration();

  //remember the initial error for rel. tol. test
  _err_feas0 = _err_feas;
  _err_optim0 = _err_optim;


  ////////////////////////////////////////////////////////////////////////////////////
  // run baby run
  ////////////////////////////////////////////////////////////////////////////////////

  // --- Algorithm status 'algStatus ----
  //-1 couldn't solve the problem (most likely because small search step. Restauration phase likely needed)
  // 0 stopped due to tolerances, including acceptable tolerance, or relative tolerance
  // 1 max iter reached
  // 2 user stop via the iteration callback

  //int algStatus=0;
  bool bret=true;
  _solverStatus = NlpSolve_Pending;
  while(notConverged) {


    //TODO: this signature does not make sense for the AL
    //user callback
    //if(!nlp->user_callback_iterate(_iter_num, _f_nlp,
	//			   *_it_curr->get_x(),
	//			   *_it_curr->get_zl(),
	//			   *_it_curr->get_zu(),
	//			   *_penaltyFcn,*_d,
	//			   *_it_curr->get_yc(),  *_it_curr->get_yd(), //lambda,
	//			   _err_nlp_feas, _err_nlp_optim,
	//			   _mu,
	//			   _alpha_dual, _alpha_primal,  lsNum)) {
    //  _solverStatus = User_Stopped; break;
    //}

    
    /****************************************************
     * Solve the AL subproblem
     ***************************************************/
    subproblemSolver.setTolerance(_eps_tol_optim);
    nlp->log->printf(hovScalars, "AugLagrSolver[%d]: Subproblem tolerance is set to %.5e\n", _iter_num+1, _eps_tol_optim);
    nlp->log->printf(hovScalars, "AugLagrSolver[%d]: Subproblem feasibility is set to %.5e\n", _iter_num+1, _eps_tol_feas);
    
    nlp->runStats.tmSolverInternal.start();  
    subproblemSolver.run(); 
    nlp->runStats.tmSolverInternal.stop();

    subproblemSolver.getSolution(_it_curr->local_data());
    subproblemSolver.getSolution_duals(_zL->local_data(), _zU->local_data());

    std::string name = "iter" + std::to_string(_iter_num+1) + ".txt";
    FILE *f3=fopen(name.c_str(),"w");
    _it_curr->print(f3);
    fclose(f3);

    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", _iter_num);
    nlp->log->write("", *_it_curr, hovIteration);
    nlp->log->write("", *_lam_curr, hovIteration);

    //this code is here because we want to reuse lambda from previous iteration
    //otherwise the QN switch is completely handled in the subproblem
    //we need SWITCH_IT -1, we are setting things up for the next subproblem
    // const int SWITCH_IT = nlp->options->GetInteger("quasi_newton_switch_it");
    // if (SWITCH_IT > 0 && _iter_num > SWITCH_IT-1)
    // {
    //     nlp->log->printf(hovWarning, "Solver: switching to the Quasi-Newton mode in the next iteration!\n");
    //     nlp->set_starting_point(n_vars, _it_curr->local_data_const());
        
    //     continue;//skip the lambda update
    // }

    /*************************************************
     * NLP Problem and Error evalutaions
     ************************************************/
    //TODO: probably doesn't need to re-evaluate Ipopt's fcns
    evalNlp(_it_curr, _f_nlp);
    evalNlpErrors(_it_curr, _zL, _zU, residual, _err_feas, _err_optim);
    
    nlp->log->printf(hovScalars, "AugLagrSolver[%d]: Subproblem optimality error %.5e\n", _iter_num+1, _err_optim);
    nlp->log->printf(hovScalars, "AugLagrSolver[%d]: Subproblem feasibility error is %.5e\n", _iter_num+1, _err_feas);
   
    name = "resid" + std::to_string(_iter_num+1) + ".txt";
    FILE *f33=fopen(name.c_str(),"w");
    residual->print(f33);
    fclose(f33);

    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", _iter_num);
    nlp->log->write("", *residual, hovIteration);
    
    /*************************************************
     *  Convergence check
     ************************************************/
    _iter_num++; nlp->runStats.nIter=_iter_num;
    outputIteration();
    if(checkTermination(_err_feas, _err_optim, _iter_num, _solverStatus)) {
        break; //terminate the loop
    }

    /************************************************
     * update rho and lambdas
     ************************************************/
    if (_err_feas <= _eps_tol_feas)
    {
        //check termination conditions   
        if(checkTermination(_err_feas, _err_optim, _iter_num, _solverStatus)) {break;}

        // update multipliers
        nlp->log->printf(hovScalars, "AugLagrSolver: Updating Lagrange multipliers.\n");
        updateLambda();

        //tighten tolerances
        _alpha = 1./_rho_curr;
        //_eps_tol_optim *= _alpha;
        //_eps_tol_feas *= pow(_alpha, 0.9);
        _eps_tol_optim *= 0.5;
        _eps_tol_feas *= 0.9;
        
    }
    else
    {
        //increase penalty parameter
        nlp->log->printf(hovScalars, "AugLagrSolver: Increasing penalty parameter.\n");
        updateRho();

        // tighten tolerances
        _alpha = (1./_rho_curr);
        _eps_tol_optim = _eps_tol_optim0 * _alpha;
        _eps_tol_feas = 0.9*_eps_tol_feas0 * pow(_alpha, 0.1);
    }


    /************************************************
     * set starting point for the next major iteration
     ************************************************/
    nlp->set_starting_point(n_vars, _it_curr->local_data_const());

  }

  nlp->runStats.tmOptimizTotal.stop();

  //_solverStatus contains the termination information
  //TODO: displayTerminationMsg();

  //TODO: user callback signature does not make sense
  //user callback
  //nlp->user_callback_solution(_solverStatus,
  //  		      *_it_curr->get_x(),
  //  		      *_it_curr->get_zl(),
  //  		      *_it_curr->get_zu(),
  //  		      *_penaltyFcn,*_d,
  //  		      *_it_curr->get_yc(),  *_it_curr->get_yd(),
  //  		      _f_nlp);

  return _solverStatus;
}


//TODO: can be merged with evalNlpErrors()
bool hiopAugLagrSolver::evalNlp(hiopVectorPar* iter,                              
                               double &f/**, hiopVector& c_, hiopVector& d_, 
                               hiopVector& gradf_,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d*/)
{
  bool new_x=true, bret; 
  double* x = iter->local_data();//local_data_const();
  //hiopVectorPar 
  //  &c=dynamic_cast<hiopVectorPar&>(c_), 
  //  &d=dynamic_cast<hiopVectorPar&>(d_), 
  //  &gradf=dynamic_cast<hiopVectorPar&>(gradf_);
  
  //we want the original objective, not the AL!!
  bret = nlp->eval_f_user(n_vars, x, new_x, f); assert(bret);

  new_x= false; //same x for the rest
  //bret = nlp->eval_grad_f(x, new_x, gradf.local_data());  assert(bret);
  //bret = nlp->eval_c     (x, new_x, c.local_data());     assert(bret);
  //bret = nlp->eval_d     (x, new_x, d.local_data());     assert(bret);
  //bret = nlp->eval_Jac_c (x, new_x, Jac_c.local_data()); assert(bret);
  //bret = nlp->eval_Jac_d (x, new_x, Jac_d.local_data()); assert(bret);

  return bret;
}


/**
 * Evaluates errors  of the Augmented lagrangian, namely
 * the feasibility error represented by the penalty function p(x,s)
 * and the optimality error represented by gradient of the Lagrangian
 * d_L = d_f(x) - J(x)^T lam
 *
 * @param[in] current_iterate The latest iterate in (x,s)
 * @param[out] resid Residual class keeping information about the NLP errors
 * @param[out} err_optim, err_feas Optimality and feasibility errors
 */
bool hiopAugLagrSolver::evalNlpErrors(const hiopVector *current_iterate, 
                                      const hiopVector *zL, const hiopVector *zU, hiopResidualAugLagr *resid, 
                                      double& err_feas, double& err_optim)
{
  double *penaltyFcn = resid->getFeasibilityPtr();
  double *gradLagr = resid->getOptimalityPtr();
  const double *_it_curr_data = _it_curr->local_data_const();
  const double *_zL_data = _zL->local_data_const();
  const double *_zU_data = _zU->local_data_const();
  bool new_x = true;

  //evaluate the AugLagr penalty fcn and gradient of the Lagrangian
  bool bret = nlp->eval_residuals(n_vars, _it_curr_data, _zL_data, _zU_data, new_x, penaltyFcn, gradLagr);
  assert(bret);
  
  //recompute the residuals norms
  resid->update();
  
  //actual nlp errors 
  err_feas  = resid->getFeasibilityNorm();
  err_optim = resid->getOptimalityNorm();

  return bret;
}


/**
 * Test checking for stopping the augmented Lagrangian loop given the NLP errors in @resid, number of iterations @_iter_num. Sets the status if appropriate.
 */
bool hiopAugLagrSolver::checkTermination(double err_feas, double err_optim, const int iter_num, hiopSolveStatus &status)
{
  if (err_feas<=_EPS_TOL && err_optim<=_EPS_TOL)
  {
      status = Solve_Success;
      return true;
  }
  
  if (iter_num>=_MAX_N_IT)
  {
      status = Max_Iter_Exceeded;
      return true;
  }

  // if(_EPS_RTOL>0) {
  //   if(err_optim   <= _EPS_RTOL * _err_optim0 &&
  //      err_feas    <= _EPS_RTOL * _err_feas0)
  //   {
  //     status = Solve_Success_RelTol;
  //     return true;
  //   }
  // }

  // if (err_feas<=_EPS_TOL_ACCEP && err_optim<=_EPS_TOL_ACCEP) _n_accep_iters++;
  // else _n_accep_iters = 0;

  // if(_n_accep_iters>=_ACCEP_N_IT) { status = Solve_Acceptable_Level; return true; }

  return false;
}

void hiopAugLagrSolver::outputIteration()
{
  if(_iter_num/10*10==_iter_num) 
    nlp->log->printf(hovSummary, "iter    objective     inf_pr     inf_du  ||dLam||  lg(rho)  iter_inner\n");

    if (_iter_num > 0)
    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e %7.3e %7.3e %6.2f %8d\n",
                     _iter_num, _f_nlp, _err_feas, _err_optim, _nrm_dlam, log10(_rho_curr), subproblemSolver.getNumIterations()); 
    else
    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e %7.3e %7.3e %6.2f %s\n",
                     _iter_num, _f_nlp, _err_feas, _err_optim, _nrm_dlam, log10(_rho_curr), "      --"); 
}

/**
 * Computes new value of the lagrange multipliers estimate
 * lam_k+1 = lam_k - (penaltyFcn_k * rho_k)
 * 
 * saves the 2-norm of the update (penaltyFcn_k * rho_k) in #_nrm_dlam
 */
void hiopAugLagrSolver::updateLambda()
{
    double *_lam_data = _lam_curr->local_data();
    const double *penaltyFcn = residual->getFeasibilityPtr();

    // compute new value of the multipliers
    double dlam2 = 0.;
    for (long long i=0; i<m_cons; i++)
    {
       const double dlam = penaltyFcn[i] * _rho_curr;
        _lam_data[i] -= dlam;
        dlam2 += dlam*dlam;
    }

    //update the multipliers in the adapter class
    nlp->set_lambda(_lam_curr);

    //2-norm of the lambda update
    _nrm_dlam = sqrt(dlam2);
}


/**
 * Computes new value of the penalty parameter
 */
void hiopAugLagrSolver::updateRho()
{
    _nrm_dlam = 0.;
    if (_rho_curr > 1e10) return;

    //compute new value of the penalty parameter
    _rho_curr = 5.0*_rho_curr;

    //update the penalty parameter in the adapter class
    nlp->set_rho(_rho_curr);
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAugLagrSolver::getObjective() const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    nlp->log->printf(hovError, "getObjective: hiOp did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    nlp->log->printf(hovWarning, "getObjective: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  return _f_nlp;
}

/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAugLagrSolver::getSolution(double* x) const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    nlp->log->printf(hovError, "getSolution: hiOp did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    nlp->log->printf(hovWarning, "getSolution: hiOp have not completed yet. The primal vector returned may not be optimal.");

  memcpy(x, _it_curr->local_data_const(), n_vars*sizeof(double));
  //TODO: this copies also the slacks, we probably don't want those
}

/* returns the status of the solver */
hiopSolveStatus hiopAugLagrSolver::getSolveStatus() const
{
  return _solverStatus;
}

/* returns the number of iterations */
int hiopAugLagrSolver::getNumIterations() const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    nlp->log->printf(hovError, "getNumIterations: hiOp did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    nlp->log->printf(hovWarning, "getNumIterations: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  
  return nlp->runStats.nIter;
}
}

