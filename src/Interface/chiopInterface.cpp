#include "chiopInterface.hpp"
extern "C" {

using namespace hiop;

// These are default options for the C interface for now. Setting options from C will be added in the future.
int hiop_createProblem(cHiopProblem *prob) {
  cppUserProblem * cppproblem = new cppUserProblem(prob);
  hiopNlpMDS *nlp = new hiopNlpMDS(*cppproblem);
  nlp->options->SetStringValue("duals_update_type", "linear");
  nlp->options->SetStringValue("duals_init", "zero");

  nlp->options->SetStringValue("Hessian", "analytical_exact");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("compute_mode", "hybrid");

  nlp->options->SetIntegerValue("verbosity_level", 3);
  nlp->options->SetNumericValue("mu0", 1e-1);
  prob->refcppHiop = nlp;
  prob->hiopinterface = cppproblem;
  return 0;
} 

int hiop_solveProblem(cHiopProblem *prob) {
  hiopSolveStatus status;
  hiopAlgFilterIPMNewton solver(prob->refcppHiop);
  status = solver.run();
  prob->obj_value = solver.getObjective();
  solver.getSolution(prob->solution);
  return 0;
}

int hiop_destroyProblem(cHiopProblem *prob) {
  delete prob->refcppHiop;
  delete prob->hiopinterface;
  return 0;
}
} // extern C
