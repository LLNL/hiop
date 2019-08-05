#include "hiopAugLagrSubproblem.hpp"

namespace hiop
{
hiopAugLagrSubproblem::hiopAugLagrSubproblem(hiopAugLagrNlpAdapter *nlp_AugLagr_in):
  subproblem(nlp_AugLagr_in),
  ipoptApp(nullptr),
  hiopSolver(nullptr),
  hiopWrapper(nullptr),
  _subproblemStatus(NotInitialized),
  _solverStatus(NlpSolve_IncompleteInit),
  _EPS_TOL_OPTIM(1e-6)
{
}

hiopAugLagrSubproblem::~hiopAugLagrSubproblem()
{
  if (ipoptApp)   delete ipoptApp;
  if (hiopSolver) delete hiopSolver;
  if (hiopWrapper) delete hiopWrapper;
}

/** Tolerance set by the main Aug Lagr solver, the subproblem
 * tolerance is gradually tightened as the solution is approached
 * in outer AL iterations. */
void hiopAugLagrSubproblem::setTolerance(double tol)
{
  _EPS_TOL_OPTIM = tol;
}

/** Initializes the subproblem solver and keeps the solver so that it is ready
 *  to be used also for the following subproblem solutions. This prevents
 *  the creation/destruction of the subproblem solver in every major iteration.
 *  We also assume that the major iterations may be restarted with different
 *  solver, or run multiple times so we keep all the previously created solvers */
void hiopAugLagrSubproblem::initialize()
{
  //check subproblem options
  //_EPS_TOL_ACCEP = subproblem->options->GetNumeric("acceptable_tolerance");
  //_MAX_N_IT      = subproblem->options->GetInteger("max_iter");
  
  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
  {
    // test if ipopt was initialized previously
    if(_subproblemStatus != IpoptInitialized &&  _subproblemStatus != IpoptHiopInitialized)
    {
      //create IPOPT
      ipoptApp = IpoptApplicationFactory();

      //set IPOPT options
      //app->Options()->SetNumericValue("tol", 1e-9);
      //app->Options()->SetStringValue("mu_strategy", "adaptive");
      //app->Options()->SetStringValue("output_file", "ipopt.out");

      // Intialize the IpoptApplication and process the options
      ApplicationReturnStatus st = ipoptApp->Initialize();
      if (st != Solve_Succeeded) {
        printf("\n\n*** Error during IPOPT initialization!\n Error %d\n", st);
      }
      
      //update the Subproblem status 
      _subproblemStatus = static_cast<hiopSubproblemStatus>(
         static_cast<int>(_subproblemStatus) + 
         static_cast<int>(IpoptInitialized));
    }
  }
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
  {
    // test if hiop was initialized previously
    if(_subproblemStatus != HiopInitialized && _subproblemStatus != IpoptHiopInitialized)
    {
      //create HIOP wrapper of the problem
      hiopWrapper = new hiopNlpDenseConstraints(*subproblem); 

      // set HIOP options
      hiopWrapper->options->SetNumericValue("tolerance", _EPS_TOL_OPTIM); 
      //hiopWrapper->options->SetNumericValue("rel_tolerance", 1e-2);
      //hiopWrapper->options->SetNumericValue("acceptable_tolerance", 1e-4);
      //hiopWrapper->options->SetIntegerValue("acceptable_iterations", 10);
      //hiopWrapper->options->SetIntegerValue("max_iter", 500);
      hiopWrapper->options->SetStringValue("fixed_var", "relax"); //remove fails
      hiopWrapper->options->SetIntegerValue("verbosity_level", 0);
      //hiopWrapper->options->SetNumericValue("sigma0", 10);
      //hiopWrapper->options->SetStringValue("sigma_update_strategy",  "sigma0"); //sty, sty_inv, snrm_ynrm, sty_srnm_ynrm
      hiopWrapper->options->SetIntegerValue("secant_memory_len", 6);
      //hiopWrapper->options->SetStringValue("dualsInitialization",  "zero"); //lsq

      //create and initialize the HiopSolver
      hiopSolver = new hiopAlgFilterIPM(hiopWrapper);

      //update the Subproblem status 
      _subproblemStatus = static_cast<hiopSubproblemStatus>(
         static_cast<int>(_subproblemStatus) + 
         static_cast<int>(HiopInitialized));
    }
  }
  
  _solverStatus = NlpSolve_SolveNotCalled;
}

hiopSolveStatus hiopAugLagrSubproblem::solveSubproblem_ipopt()
{
  checkConsistency();

  // Ask Ipopt to solve the problem
  ApplicationReturnStatus st = ipoptApp->OptimizeTNLP(subproblem);
  displayTerminationMsgIpopt(st);

  if (st == Solve_Succeeded) { return Solve_Success; }
  else { return UnknownNLPSolveStatus; }

}

hiopSolveStatus hiopAugLagrSubproblem::solveSubproblem_hiop()
{
  checkConsistency();

  // Ask HiOP to solve the problem
  hiopSolveStatus st = hiopSolver->run();
  displayTerminationMsgHiop(st);

  return st;
}

/** Solves the subproblem using the specified solver */
hiopSolveStatus hiopAugLagrSubproblem::run()
{
  _solverStatus = NlpSolve_Pending;

  subproblem->log->printf(hovScalars, "Subproblem tolerance is set to %g\n", _EPS_TOL_OPTIM);

  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
    _solverStatus = solveSubproblem_ipopt();
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
    _solverStatus = solveSubproblem_hiop();
  

  return _solverStatus;
}

/** Check consistency of the current object state vs required options
 *  specified in the subproblem, i.e. if the required solver was initialized */
void hiopAugLagrSubproblem::checkConsistency()
{
  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
  {
    if (_subproblemStatus != IpoptInitialized && _subproblemStatus != IpoptHiopInitialized)
    {
      subproblem->log->printf(hovError, "hipAugLagrSubproblem consistency check: Ipopt solver not initialized, please call initialize() first!\n");
    }
  }
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
  {
    if(_subproblemStatus != HiopInitialized && _subproblemStatus != IpoptHiopInitialized)
    {
      subproblem->log->printf(hovError, "hipAugLagrSubproblem consistency check: Hiop solver not initialized, please call initialize() first!\n");
    }
  }
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAugLagrSubproblem::getObjective() const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getObjective: hiOp AugLagr subproblem did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getObjective: hiOp AL subproblem does not seem to have completed yet. The objective value returned may not be optimal.");
  
  return 100.;
  //TODO
}

/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAugLagrSubproblem::getSolution(double* x) const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getSolution: hiOp AugLagr subproblem did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getSolution: hiOp AugLagr subproblem have not completed yet. The primal vector returned may not be optimal.");

  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
  {
    //IpoptApplication doesn't provide a method to access the solution.
    //The solution is passed to user in callback finalize_solution which
    //is implemented in AugLagrNlpAdapter, the solution is thus cached there 
    subproblem->get_ipoptSolution(x);
  }
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
  {
    hiopSolver->getSolution(x);
  }

  return;
  //TODO:
  // update the current iterate, used as x0 for the next subproblem
  //hiopSolver->getSolution();
  //TODO: save also the IPM duals and do the warm start
}

/* returns the status of the solver */
hiopSolveStatus hiopAugLagrSubproblem::getSolveStatus() const
{
  return _solverStatus;
}

/* returns the number of iterations */
int hiopAugLagrSubproblem::getNumIterations() const
{
  if(_solverStatus==NlpSolve_IncompleteInit || _solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getNumIterations: hiOp did not initialize entirely or the 'run' function was not called.");
  if(_solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getNumIterations: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  
  return subproblem->runStats.nIter;
}

/***** Termination message *****/
void hiopAugLagrSubproblem::displayTerminationMsgIpopt(ApplicationReturnStatus st) {
   switch (st)  {
    case Solve_Succeeded:
      { subproblem->log->printf(hovSummary, "Successfull termination.\n%s\n", subproblem->runStats.getSummary().c_str());
        break;
      }
    case Solved_To_Acceptable_Level:
      { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Solved_To_Acceptable_Level\n");
        break;
      }
    case Infeasible_Problem_Detected:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Infeasible_Problem_Detected\n");     
        break;
      }
    case Search_Direction_Becomes_Too_Small:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Search_Direction_Becomes_Too_Small\n");      
        break;
      }
    case User_Requested_Stop:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was User_Requested_Stop\n");     
        break;
      }
    case Feasible_Point_Found:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Feasible_Point_Found\n");    
        break;
      }
    case Maximum_Iterations_Exceeded:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Maximum_Iterations_Exceeded\n");     
        break;
      }
    case Restoration_Failed:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Restoration_Failed\n");      
        break;
      }
    case Error_In_Step_Computation:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Error_In_step_Computation\n");       
        break;
      }
    case Maximum_CpuTime_Exceeded:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Maximum_CpuTime_Exceeded\n");        
        break;
      }
    case Invalid_Option:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Invalid_Options\n");  
        break;
      }
    case Invalid_Number_Detected:
       { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return code was Invalid_Number_Detected\n");         
        break;
      }
    default:
      { subproblem->log->printf(hovWarning, "hiopAugLagrSubproblem::run Ipopt return an iternal error %d\n", st); 
        break;
      }
   }
}



void hiopAugLagrSubproblem::displayTerminationMsgHiop(hiopSolveStatus st) {

  switch(st) {
  case Solve_Success: 
    {
      subproblem->log->printf(hovSummary, "Successfull termination.\n%s\n", subproblem->runStats.getSummary().c_str());
      break;
    }
  case Solve_Success_RelTol: 
    {
      subproblem->log->printf(hovSummary, "Successfull termination (error within the relative tolerance).\n%s\n", subproblem->runStats.getSummary().c_str());
      break;
    }
  case Solve_Acceptable_Level:
    {
      subproblem->log->printf(hovSummary, "Solve to only to the acceptable tolerance(s).\n%s\n", subproblem->runStats.getSummary().c_str());
      break;
    }
  case Max_Iter_Exceeded:
    {
      subproblem->log->printf(hovSummary, "Maximum number of iterations reached.\n%s\n", subproblem->runStats.getSummary().c_str());
      break;
    }
  case Steplength_Too_Small:
    {
      subproblem->log->printf(hovSummary, "Couldn't solve the problem.\n%s\n", subproblem->runStats.getSummary().c_str());
      subproblem->log->printf(hovSummary, "Linesearch returned unsuccessfully (small step). Probable cause: inaccurate gradients/Jacobians or infeasible problem.\n");
      break;
    }
  case User_Stopped:
    {
      subproblem->log->printf(hovSummary, "Stopped by the user through the user provided iterate callback.\n%s\n", subproblem->runStats.getSummary().c_str());
      break;
    }
  default:
    {
      subproblem->log->printf(hovSummary, "Do not know why hiop stopped. This shouldn't happen. :)\n%s\n", subproblem->runStats.getSummary().c_str());
      assert(false && "Do not know why hiop stopped. This shouldn't happen.");
      break;
    }
  };
}

}
