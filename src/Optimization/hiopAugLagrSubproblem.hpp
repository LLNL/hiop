#ifndef HIOP_AUGLAGR_SUBPROBLEM_HPP
#define HIOP_AUGLAGR_SUBPROBLEM_HPP

#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "IpIpoptApplication.hpp"

namespace hiop
{

enum hiopSubproblemStatus
{ ///this type needs to be contiguous
  NotInitialized       = 0, //no initialized solver
  IpoptInitialized     = 1, //only ipopt initialized
  HiopInitialized      = 2, //only hiop initialized
  IpoptHiopInitialized = 3 //both initialized (sum of initialized solvers)
};

//TODO
//we should make it look as hiopAlgFilterIPM
//and later have a common parent class

class hiopAugLagrSubproblem {
public:
  hiopAugLagrSubproblem(hiopAugLagrNlpAdapter *nlp_AugLagr_in); 
  ~hiopAugLagrSubproblem(); 

 /** Initializes the subproblem solver and keeps the solver so that it is ready
  *  to be used also for the following subproblem solutions. This prevents
  *  the creation/destruction of the subproblem solver in every major iteration.
  *  We also assume that the major iterations may be restarted with different
  *  solver, or run multiple times so we keep all the previously created solvers */
  virtual void initialize();

  /** Solves the subproblem using the specified solver */
  virtual hiopSolveStatus run();

  /* returns the objective value; valid only after 'run' method has been called */
  virtual double getObjective() const;
  /* returns the primal vector x; valid only after 'run' method has been called */
  virtual void getSolution(double* x) const;
  /* returns the duals vector zL and zU; valid only after 'run' method has been called */
  void getSolution_duals(double* zL, double* zU) const;
  /* returns the status of the solver */
  virtual hiopSolveStatus getSolveStatus() const;
  /* returns the number of iterations */
  virtual int getNumIterations() const;

public:
  void setTolerance(double tol);

private:
  hiopSolveStatus solveSubproblem_ipopt();
  hiopSolveStatus solveSubproblem_hiop();
  void checkConsistency() const;
  void displayTerminationMsgHiop(hiopSolveStatus st);
  void displayTerminationMsgIpopt(ApplicationReturnStatus st);


private:
  hiopAugLagrNlpAdapter *subproblem;    ///< instance of the AL problem with options
  IpoptApplication *ipoptApp;           ///< instance of Ipopt solver
  hiopAlgFilterIPM *hiopSolver;         ///< instance of hiop solver
  hiopNlpDenseConstraints *hiopWrapper; ///< wrapper of the subproblem for hiop

  hiopSubproblemStatus _subproblemStatus; ///< flag specifing which solver was initialized, if any
  hiopSolveStatus _solverStatus; ///< flag specifying if the solver was called or its return value

  double _EPS_TOL_OPTIM; ///< required tolerance
};

}

#endif
