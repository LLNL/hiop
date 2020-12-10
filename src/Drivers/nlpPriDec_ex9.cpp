#include "nlpPriDec_ex9.hpp"

PriDecMasterProblemEx9::PriDecMasterProblemEx9(size_t nx, size_t ny, size_t nS, size_t S)
  : nx_(nx), ny_(ny), nS_(nS), S_(S)
{
  basecase_ = new PriDecBasecaseProblemEx9(nx_);

  //todo:
  // - generate S vectors \xi (each of size nS) from U[-0.25, 0.25]
  // - create S problems PriDecRecourseProblemEx9
}
PriDecMasterProblemEx9::~PriDecMasterProblemEx9()
{
  delete basecase_;
}

hiop::hiopSolveStatus
PriDecMasterProblemEx9::solve_master(double* x,
                                     const bool& include_r,
                                     const double& rval/*=0*/, 
                                     const double* grad/*=0*/,
                                     const double*hess /*=0*/)
{
  //todo:
  //  - solve PriDecBasecaseProblemEx9
  //  - retrieve whatever is needed from it
  assert(false && "not implemented");
  return hiop::SolverInternal_Error;
}

bool PriDecMasterProblemEx9::eval_f_rterm(size_t idx, const int& n, double* x, double& rval)
{
  // todo:
  //  - set `x` for the recourse PriDecRecourseProblemEx9 corresponding to `idx`
  //  - solve the problem
  //  - retrieve obj_value
  //  - retrieve/compute the gradient 
  return false;
}

bool PriDecMasterProblemEx9::eval_grad_rterm(size_t idx, const int& n, double* x, double* grad)
{
  //todo:
  // return in grad the gradient computed in eval_f_rterm
  return false;
}
