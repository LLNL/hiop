#ifndef HIOP_EXAMPLE_EX8
#define HIOP_EXAMPLE_EX8

#include "hiopInterfacePrimalDecomp.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <cassert>
#include <cstring> //for memcpy
#include <cstdio>
#include <cmath>

using namespace hiop;

class PriDecMasterProblemEx8 : public hiopInterfacePriDecProblem
{
public:
  PriDecMasterProblemEx8(int n,
                         int S,
                         MPI_Comm comm_world=MPI_COMM_WORLD)
    : hiopInterfacePriDecProblem(comm_world),
      n_(n), S_(S)
  {

  }

  virtual ~PriDecMasterProblemEx8()
  {
  }

  hiopSolveStatus solve_master(double* x)
  {
    //pretend that the master problem has all zero solution
    for(int i=0; i<n_; i++)
      x[i] = 0.;
    return Solve_Success;
  };

  bool eval_f_rterm(size_t idx, const int& n, double* x, double& rval)
  {
    rval = 0.;
    for(int i=0; i<n; i++) rval += (x[i]-1)*(x[i]-1);
    rval *= 0.5;
    rval /= S_;
    return true;
  }
  bool eval_grad_rterm(size_t idx, const int& n, double* x, double* grad)
  {
    assert(n_ == n);
    for(int i=0; i<n; i++)
      grad[i] = (x[i]-1)/S_;
    return true;
  }  

  bool set_quadratic_regularization(/* params = ? */)
  {
    return true;
  }

  /** 
   * Returns the number S of recourse terms
   */
  size_t get_num_rterms() const
  {
    return S_;
  }
  size_t get_num_vars() const
  {
    return n_;
  }
  void get_solution(double* x) const
  {
    for(int i=0; i<n_; i++)
      x[i] = 0.;
  }

  double get_objective()
  {
    return 0.;
  }
private:
  size_t n_;
  size_t S_;

  // will need some encapsulation of the basecase NLP
  // nlpMDSForm_ex4.hpp
};

#endif
