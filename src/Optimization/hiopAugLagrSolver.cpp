#include "hiopAugLagrSolver.hpp"
#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <algorithm>    // std::max


namespace hiop
{

hiopAugLagrSolver::hiopAugLagrSolver(NLP_CLASS_IN* nlp_in_) 
  : nlp(new hiopAugLagrNlpAdapter(nlp_in_)),
    n(0),
    m(0),
    _it_curr(nullptr),
    _lam_curr(nullptr),
    _rho_curr(100.),
    residual(nullptr),
    _err_feas0(-1.),
    _err_optim0(-1.),
    _solverStatus(NlpSolve_IncompleteInit),
    iter_num(0),
    max_n_it(1000),
    eps_tol(1e-6),
    eps_rtol(1e-6),
    eps_tol_accep(1e-4),
    accep_n_it(5),
    _n_accep_iters(0),
    rho_max(1e7)
{
  // get size of the problem and the penalty term
  long long dum1;
  nlp->get_prob_sizes(n, dum1);
  nlp->get_penalty_size(m);
  
  _it_curr  = new hiopVectorPar(n);
  _lam_curr = new hiopVectorPar(m);
  residual  = new hiopResidualAugLagr(n, m);

  reloadOptions();
  reInitializeNlpObjects();
  resetSolverStatus();
}

hiopAugLagrSolver::~hiopAugLagrSolver() 
{
    if(nlp)       delete nlp;
    if(_it_curr)  delete _it_curr;
    if(_lam_curr) delete _lam_curr;
    if(residual)     delete residual;
}



/**
 */
hiopSolveStatus hiopAugLagrSolver::run()
{

    //TODO implement logging in log
  //nlp->log->printf(hovSummary, "===============\nHiop AugLagr SOLVER\n===============\n");

  reloadOptions();
  reInitializeNlpObjects();
  resetSolverStatus();

  _solverStatus = NlpSolve_SolveNotCalled;
  
  nlp->runStats.initialize();
  nlp->runStats.tmOptimizTotal.start();
  nlp->runStats.tmStartingPoint.start();

  //initialize curr_iter by calling TNLP starting point + do something about slacks
  // and set starting point at the Adapter class for the first major AL iteration
  nlp->get_user_starting_point(n, _it_curr->local_data());
  nlp->set_starting_point(n, _it_curr->local_data_const());

  //set initial guess of the multipliers and the penalty parameter
  //TODO hot start for lambda
  _lam_curr->setToConstant(1.); nlp->set_lambda(_lam_curr);
  _rho_curr = 100.;             nlp->set_rho(_rho_curr);
  
  nlp->runStats.tmStartingPoint.stop();

  //evaluate the residuals
  bool new_x = true;
  evalNlpErrors(_it_curr, new_x, residual);
  //nlp->log->write("First residual-------------", *residual, hovIteration);

  //check termination conditions   
  bool notConverged = true;
  if(checkTermination(residual, iter_num, _solverStatus)) {
      notConverged = false;
    }

  //additional initializations
  iter_num=0; nlp->runStats.nIter=iter_num;
  _err_feas0 = residual->getFeasibilityNorm();
  _err_optim0 = residual->getOptimalityNorm();


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


    //nlp->log->printf(hovScalars, "  Nlp    errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n", _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
    //nlp->log->printf(hovScalars, "  LogBar errs: pr-infeas:%20.14e   dual-infeas:%20.14e  comp:%20.14e  overall:%20.14e\n",_err_log_feas, _err_log_optim, _err_log_complem, _err_log);
    //outputIteration(lsStatus, lsNum);

    //TODO: this signature does not make sense for the AL
    //user callback
    //if(!nlp->user_callback_iterate(iter_num, _f_nlp,
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
     * Solve the AL subproblem by calling HiOP or Ipopt
     ***************************************************/
    nlp->runStats.tmSolverInternal.start();  
    hiopNlpDenseConstraints subproblem(*nlp);

    //subproblem.options->SetIntegerValue("verbosity_level", 4);
    //subproblem.options->SetNumericValue("tolerance", 1e-4);
    //subproblem.options->SetStringValue("dualsInitialization",  "zero");
    //subproblem.options->SetIntegerValue("max_iter", 2);

    hiopAlgFilterIPM solver(&subproblem);
    hiopSolveStatus status = solver.run();

    nlp->runStats.tmSolverInternal.stop();
    //solver.getObjective();
    
    // update the current iterate, used as x0 for the next subproblem
    solver.getSolution(_it_curr->local_data());
    
    
    //nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", iter_num); nlp->log->write("", *_it_curr, hovIteration);

    /*************************************************
     * Error evalutaion & Termination check
     ************************************************/
    //evaluate the residuals and NLP errors
    //TODO: new_x = false if hiop/IPOPT evaluates fcn/jac at the solution
    //new_x = true if fcn/Jac is not evaluated at the solution
    evalNlpErrors(_it_curr, new_x, residual);
    //nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num); nlp->log->write("", *residual, hovIteration);
  
    //check termination conditions   
    if(checkTermination(residual, iter_num, _solverStatus)) {
        break;
      }

    /************************************************
     * set starting point for the next major iteration
     ************************************************/
    nlp->set_starting_point(n, _it_curr->local_data_const());

    /************************************************
     * update rho and lambdas
     ************************************************/
    updateLambda();
    updateRho();

    iter_num++; nlp->runStats.nIter=iter_num;
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

/**
 * Evaluates termination criteria of the Augmented lagrangian, namely
 * the feasibility error represented by the penalty function p(x,s)
 * and the optimality error represented by gradient of the Lagrangian
 * d_L = d_f(x) + J(x)^T lam
 *
 * @param[in] current_iterate The latest iterate in (x,s)
 * @param[in] new_x
 * @param[out] resid Residual class keeping information about the NLP errors
 */
bool hiopAugLagrSolver::evalNlpErrors(const hiopVector *current_iterate, bool new_x,
                                                hiopResidualAugLagr *resid)
{

    double *penaltyFcn = resid->getFeasibilityPtr();
    double *gradLagr = resid->getOptimalityPtr();
    const double *_it_curr_data = _it_curr->local_data_const();

    //evaluate the Adapter penalty fcn and gradient of the Lagrangian
    bool bret = nlp->eval_residuals(n, _it_curr_data, new_x, penaltyFcn, gradLagr);
    assert(bret);
  
    //recompute the residuals norms
    resid->update();

    return bret;
}


/**
 * Test checking for stopping the augmented Lagrangian loop given the NLP errors in @resid, number of iterations @iter_num. Sets the status if appropriate.
 */
bool hiopAugLagrSolver::checkTermination(const hiopResidualAugLagr *resid, const int iter_num, hiopSolveStatus &status)
{
  const double err_feas = resid->getFeasibilityNorm();
  const double err_optim = resid->getOptimalityNorm();

  if (err_feas<=eps_tol && err_optim<=eps_tol)
  {
      status = Solve_Success;
      return true;
  }
  
  if (iter_num>=max_n_it)
  {
      status = Max_Iter_Exceeded;
      return true;
  }

  if(eps_rtol>0) {
    if(err_optim   <= eps_rtol * _err_optim0 &&
       err_feas    <= eps_rtol * _err_feas0)
    {
      status = Solve_Success_RelTol;
      return true;
    }
  }

  if (err_feas<=eps_tol_accep && err_optim<=eps_tol_accep) _n_accep_iters++;
  else _n_accep_iters = 0;

  if(_n_accep_iters>=accep_n_it) { status = Solve_Acceptable_Level; return true; }

  return false;
}

/**
 * Computes new value of the lagrange multipliers estimate
 * lam_k+1 = lam_k + penaltyFcn_k/rho_k
 */
void hiopAugLagrSolver::updateLambda()
{
    double *_lam_data = _lam_curr->local_data();
    const double *penaltyFcn = residual->getFeasibilityPtr();

    // compute new value of the multipliers
    for (long long i=0; i<m; i++)
        _lam_data[i] += penaltyFcn[i]/_rho_curr;

    //update the multipliers in the adapter class
    nlp->set_lambda(_lam_curr);
}


/**
 * Computes new value of the penalty parameter
 */
void hiopAugLagrSolver::updateRho()
{
    //compute new value of the penalty parameter
    _rho_curr = std::max(10*_rho_curr, rho_max); //TODO

    //update the penalty parameter in the adapter class
    nlp->set_rho(_rho_curr);
}

void hiopAugLagrSolver::reInitializeNlpObjects()
{
}

void hiopAugLagrSolver::reloadOptions()
{
  //algorithm parameters parameters
  eps_tol  = nlp->options->GetNumeric("tolerance");        //absolute error for the nlp
  eps_rtol = nlp->options->GetNumeric("rel_tolerance");    //relative error (to errors for the initial point)
  eps_tol_accep = nlp->options->GetNumeric("acceptable_tolerance");

  max_n_it  = nlp->options->GetInteger("max_iter");
  accep_n_it    = nlp->options->GetInteger("acceptable_iterations");
}

void hiopAugLagrSolver::resetSolverStatus()
{
  _n_accep_iters = 0;
  _solverStatus = NlpSolve_IncompleteInit;
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAugLagrSolver::getObjective() const
{
//  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
//    nlp->log->printf(hovError, "getObjective: hiOp did not initialize entirely or the 'run' function was not called.");
//  if(_solverStatus==NlpSolve_Pending)
//    nlp->log->printf(hovWarning, "getObjective: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
//  return nlp->user_obj(_f_nlp);
  return 0.0; //TODO
}

/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAugLagrSolver::getSolution(double* x) const
{
//  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
//    nlp->log->printf(hovError, "getSolution: hiOp did not initialize entirely or the 'run' function was not called.");
//  if(_solverStatus==NlpSolve_Pending)
//    nlp->log->printf(hovWarning, "getSolution: hiOp have not completed yet. The primal vector returned may not be optimal.");
//
//  hiopVectorPar& it_x = dynamic_cast<hiopVectorPar&>(*it_curr->get_x());
//  //it_curr->get_x()->copyTo(x);
//  nlp->user_x(it_x, x);
//TODO
}

/* returns the status of the solver */
hiopSolveStatus hiopAugLagrSolver::getSolveStatus() const
{
  return _solverStatus;
}

/* returns the number of iterations */
int hiopAugLagrSolver::getNumIterations() const
{
//TODO
//  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
//    nlp->log->printf(hovError, "getNumIterations: hiOp did not initialize entirely or the 'run' function was not called.");
//  if(_solverStatus==NlpSolve_Pending)
//    nlp->log->printf(hovWarning, "getNumIterations: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  return nlp->runStats.nIter;
}
}

