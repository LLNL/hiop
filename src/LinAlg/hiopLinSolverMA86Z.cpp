#include "hiopLinSolverMA86Z.hpp"

#include "blasdefs.hpp"

namespace hiop
{
  hiopLinSolverMA86Z::hiopLinSolverMA86Z(int n_, hiopNlpFormulation* nlp_/*=NULL*/)
    : hiopLinSolver(), n(n_)
  {
    nlp = nlp_;
  }
  
  hiopLinSolverMA86Z::~hiopLinSolverMA86Z()
  {

  }

  int hiopLinSolverMA86Z::matrixChanged()
  {
    return -1;
  }

  void hiopLinSolverMA86Z::solve(hiopVector& x)
  {
    
  }

  void hiopLinSolverMA86Z::solve(hiopMatrix& X)
  {
    
  }

} //end namespace hiop
