#include "hiopAugLagrSubproblem.hpp"

namespace hiop
{
hiopAugLagrSubproblem::hiopAugLagrSubproblem(hiopAugLagrNlpAdapter *nlp_AugLagr_in):
  subproblem(nlp_AugLagr_in),
  ipoptApp(nullptr),
  hiopSolver(nullptr),
  hiopWrapper(nullptr),
  subproblemStatus(NotInitialized),
  solverStatus(NlpSolve_IncompleteInit),
  _EPS_TOL_OPTIM(1e-6)
{
}

hiopAugLagrSubproblem::~hiopAugLagrSubproblem()
{
  if (ipoptApp)   delete ipoptApp;
  if (hiopSolver) delete hiopSolver;
  if (hiopWrapper) delete hiopWrapper;
}

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
    if(subproblemStatus != IpoptInitialized &&  subproblemStatus != IpoptHiopInitialized)
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
      subproblemStatus = static_cast<hiopSubproblemStatus>(
         static_cast<int>(subproblemStatus) + 
         static_cast<int>(IpoptInitialized));
    }
  }
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
  {
    // test if hiop was initialized previously
    if(subproblemStatus != HiopInitialized && subproblemStatus != IpoptHiopInitialized)
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
      subproblemStatus = static_cast<hiopSubproblemStatus>(
         static_cast<int>(subproblemStatus) + 
         static_cast<int>(HiopInitialized));
    }
  }
  
  solverStatus = NlpSolve_SolveNotCalled;
}

hiopSolveStatus hiopAugLagrSubproblem::solveSubproblem_ipopt()
{
  checkConsistency();

  // Ask Ipopt to solve the problem
  ApplicationReturnStatus st = ipoptApp->OptimizeTNLP(subproblem);

  if (st == Solve_Succeeded) {
    printf("\n\n*** The subproblem solved!\n");
  }
  else {
    printf("\n\n*** The subproblem FAILED!\nError %d\n", st);
    exit(1);
  }

 return Solve_Success;
}

hiopSolveStatus hiopAugLagrSubproblem::solveSubproblem_hiop()
{
  checkConsistency();

  // Ask HiOP to solve the problem
  hiopSolveStatus st = hiopSolver->run();

  if (st == Solve_Success) {
    printf("\n\n*** The subproblem solved!\n");
  }
  else {
    printf("\n\n*** The subproblem FAILED!\nError %d\n", st);
    exit(1);
  }

  return st;
}

/** Solves the subproblem using the specified solver */
hiopSolveStatus hiopAugLagrSubproblem::run()
{
  solverStatus = NlpSolve_Pending;

  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
    solverStatus = solveSubproblem_ipopt();
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
    solverStatus = solveSubproblem_hiop();

  return solverStatus;
}

/** Check consistency of the current object state vs required options
 *  specified in the subproblem, i.e. if the required solver was initialized */
void hiopAugLagrSubproblem::checkConsistency()
{
  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
  {
    if (subproblemStatus != IpoptInitialized && subproblemStatus != IpoptHiopInitialized)
    {
      printf("\n\n*** Ipopt solver not initialized, please call initialize() first!\n");
      exit(1);
    }
  }
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
  {
    if(subproblemStatus != HiopInitialized && subproblemStatus != IpoptHiopInitialized)
    {
      printf("\n\n*** Hiop solver not initialized, please call initialize() first!\n");
      exit(1);
    }
  }
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAugLagrSubproblem::getObjective() const
{
  if(solverStatus==NlpSolve_IncompleteInit || solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getObjective: hiOp did not initialize entirely or the 'run' function was not called.");
  if(solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getObjective: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  
  return 100.;;
  //TODO
}

/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAugLagrSubproblem::getSolution(double* x) const
{
  if(solverStatus==NlpSolve_IncompleteInit || solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getSolution: hiOp did not initialize entirely or the 'run' function was not called.");
  if(solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getSolution: hiOp have not completed yet. The primal vector returned may not be optimal.");

  if(subproblem->options->GetString("subproblem_solver")=="ipopt")
    return;
  else if(subproblem->options->GetString("subproblem_solver")=="hiop")
    return;
  //TODO:
  // update the current iterate, used as x0 for the next subproblem
  //hiopSolver->getSolution();
  //TODO: save also the IPM duals and do the warm start
}

/* returns the status of the solver */
hiopSolveStatus hiopAugLagrSubproblem::getSolveStatus() const
{
  return solverStatus;
}

/* returns the number of iterations */
int hiopAugLagrSubproblem::getNumIterations() const
{
  if(solverStatus==NlpSolve_IncompleteInit || solverStatus == NlpSolve_SolveNotCalled)
    subproblem->log->printf(hovError, "getNumIterations: hiOp did not initialize entirely or the 'run' function was not called.");
  if(solverStatus==NlpSolve_Pending)
    subproblem->log->printf(hovWarning, "getNumIterations: hiOp does not seem to have completed yet. The objective value returned may not be optimal.");
  
  return subproblem->runStats.nIter;
}
}
