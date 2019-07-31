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
  bool evalNlp(hiopVectorPar* iter,                              
        double &f /**, hiopVector& c_, hiopVector& d_, 
        hiopVector& gradf_,  hiopMatrixDense& Jac_c,  hiopMatrixDense& Jac_d*/);
 
  /** Evaluates  errors, i.e. feasibility and optimality */
  bool evalNlpErrors(const hiopVector *current_iterate, 
        hiopResidualAugLagr *resid, double& err_feas, double& err_optim);
  
  /** Performs test whether the termination criteria are satisfied */
  bool checkTermination(double err_feas, double err_optim,
                const int iter_num, hiopSolveStatus &status);
  
  void outputIteration();
  
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
  long long n_vars; ///< number of variables
  long long m_cons; ///< number of penalty terms
  hiopVectorPar *_it_curr;  ///< curent iterate (x,s)
  hiopVectorPar *_lam_curr; ///< current estimate of the multipliers
  double _rho_curr;         ///< current value of the penalty

  //feasibility and optimality residuals
  hiopResidualAugLagr *residual; ///< residual norms 

  //internal flags related to the state of the solver
  hiopSolveStatus _solverStatus;
  int _iter_num;        ///< iteration number
  int _n_accep_iters;  ///< number of encountered consecutive acceptable iterates
  double _f_nlp;       ///< current objective function value
  double _err_feas0, _err_optim0; ///< initial errors
  double _err_feas, _err_optim; ///< current errors

  //user options and parameters
  double _LAMBDA0;       ///< initial value of the multipliers
  double _RHO0;          ///< initial value of the penalty parameter
  double _EPS_TOL;       ///< abs tolerance for the NLP error (same for feas and optim)
  double _EPS_RTOL;      ///< rel tolerance for the NLP error (same for feas and optim)
  double _EPS_TOL_ACCEP; ///< acceptable tolerance (required at _ACCEP_N_IT iterations)
  int _MAX_N_IT;        ///< maximum number of iterations
  int _ACCEP_N_IT;      ///< acceptable number of iterations
  
  // internal algorithm options
  double _alpha; ///< positive constant
  double _eps_tol_feas; ///< required feasibility of the subproblem
  double _eps_tol_optim; ///< required optimality tolerance of the subproblem
};

}

#endif
