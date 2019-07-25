#ifndef HIOP_AUGLAGR_SOLVER_HPP
#define HIOP_AUGLAGR_SOLVER_HPP


#include "hiopResidualAugLagr.hpp"

namespace hiop
{

class hiopAugLagrSolver {

public:
  hiopAugLagrSolver(NLP_CLASS_IN* nlp_in_); 
  ~hiopAugLagrSolver(); 

  //TODO
  //we should make it look as hiopAlgFilterIPM
  //and later have a common parent class
  
  virtual hiopSolveStatus run();

  /* returns the objective value; valid only after 'run' method has been called */
  virtual double getObjective() const;
  /* returns the primal vector x; valid only after 'run' method has been called */
  virtual void getSolution(double* x) const;
  /* returns the status of the solver */
  virtual hiopSolveStatus getSolveStatus() const;
  /* returns the number of iterations */
  virtual int getNumIterations() const;

private:
  /** Evaluates termination criteria, i.e. feasibility and optimality */
  bool evalNlpErrors(const hiopVector *current_iterate, bool new_x,
                     hiopResidualAugLagr *resid);
  /** Performs test whether the termination criteria are satisfied */
  bool checkTermination(const hiopResidualAugLagr *resid, const int iter_num,
                        hiopSolveStatus &status);
  /** Update strategies for the multipliers and penalty */
  void updateLambda();
  void updateRho();

  /** Solver state management */
  void reloadOptions();
  void resetSolverStatus();
  void reInitializeNlpObjects();

protected:
  
  hiopAugLagrNlpAdapter* nlp; ///< Representation of the Aug Lagr. problem

  //Augmented Lagrangian problem variables
  long long n; ///< number of variables
  long long m; ///< number of penalty terms
  hiopVectorPar *_it_curr;  ///< curent iterate (x,s)
  hiopVectorPar *_lam_curr; ///< current estimate of the multipliers
  double _rho_curr;         ///< current value of the penalty

  //feasibility and optimality residuals
  hiopResidualAugLagr *residual; ///< residual norms 

  //errors
  double _err_feas0, _err_optim0; ///< initial error

  //internal flags related to the state of the solver
  hiopSolveStatus _solverStatus;
  
  int iter_num;        ///< iteration number
  int _n_accep_iters;  ///< number of encountered consecutive acceptable iterates

  //options and parameters //TODO use uppercase for the scalar constants
  double eps_tol;       ///< abs tolerance for the NLP error
  double eps_rtol;      ///< rel tolerance for the NLP error
  double eps_tol_accep; ///< acceptable tolerance (required at accep_n_it iterations)
  int accep_n_it;      ///< acceptable number of iterations
  int max_n_it;        ///< maximum number of iterations
  double rho_max;      ///< max value of the penalty parameter
};

}

#endif
