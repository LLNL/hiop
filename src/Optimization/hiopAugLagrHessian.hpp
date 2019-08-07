#ifndef HIOP_AUGLAGR_HESSIAN
#define HIOP_AUGLAGR_HESSIAN

#include "hiop_defs.hpp"

#include "hiopVector.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopAugLagrNlpAdapter.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 

#ifndef MPI_COMM
#define MPI_Comm int
#endif
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF 0
#endif
#include <cstddef>

#endif 

#include <cstdio>
#include <vector> 

using std::vector;

namespace hiop
{

class hiopVector;
class hiopVectorPar;

class hiopAugLagrHessian //: public hiopMatrixSparse
{
public:
  hiopAugLagrHessian(NLP_CLASS_IN *nlp_in_, int n_vars, int m_cons, int nnz);
  ~hiopAugLagrHessian();
 
  int nnz();
  void assemble(const double *x, bool new_x, const hiopVectorPar &lambda, const double rho,
        const hiopVectorPar &penaltyFcn, const hiopMatrixSparse &penaltyFcn_jacobian);

private:
  /**
   *   Evaluates NLP Hessian and stores it in member #_hessianNlp.
   *   We use lambda =  2*rho*p(x) - lambda in order to account for
   *   contribution not only of the Lagrangian term but also the penalty term.
   *   _hessianNlp = hess_obj + sum_i lambda*H_i,
   *   where H_i are the penalty function Hessians.
   *   The sparse structure is initialized during the first call
   */
  bool eval_hess_nlp(const double *x_in, bool new_x, const hiopVectorPar &lambda, const double rho, const hiopVectorPar &penaltyFcn);

  /** C = alpha * A' * A + beta*B
  */
  void transAAplusB(hiopMatrixSparse &C, double alpha, const hiopMatrixSparse &A, double beta, const hiopMatrixSparse &B);


#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  //bool assertSymmetry(double tol=1e-16) const;
#endif

private:
    NLP_CLASS_IN     *nlp_in; ///< input NLP representation, neede to evaluate NLP hessian
    int nvars_nlp;  ///<property of the input NLP
    int mcons_nlp;  ///<property of the input NLP
    int nnz_nlp;   ///<property of the input NLP
    hiopVectorPar    *_lambdaForHessEval; ///< lambda + 2*rho*c(x), used during the NLP Hessian evaluation
    hiopMatrixSparse *_hessianNlp; ///< Hessian of Lagrangian of the original NLP problem evaluated with extended #_lambdaForHessEval
    hiopMatrixSparse *_hessianAugLagr; ///< Hessian of the AL problem
};


}
#endif
