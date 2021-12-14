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
  hiopAugLagrHessian(NLP_CLASS_IN *nlp_in_, int n_vars, int n_slacks, int m_cons, int nnz);
  ~hiopAugLagrHessian();
 
  int nnz();
  void assemble(const double *x, bool new_x, double obj_factor, const hiopVectorPar &lambda, const double rho,
        const hiopVectorPar &penaltyFcn, const hiopMatrixSparse &penaltyFcn_jacobian, long long *cons_ineq_mapping);
  void getStructure(int *iRow, int *jCol);
  void getValues(double *values);

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
  * The method can work with #C being either an empty sparse matrix,
    i.e. hiopMatrixSparse(0,0,0.), in which case the storage is allocated
    and the sparse structure is created using vvCols and vvValues. In case
    #C already contains all the required storage space, we only update the numerical values
    of the nonzeros (assuming that the structure was set up previously).
    
    \param[out] C The method computes and returns only the lower triangular part of the symmetric result.
    \param[out] vvCols, vvValues The method computes and returns only the lower triangular part of the symmetric result.
    \param[in] structureNotInitialized Switch deciding which output will be updated, either C or vvCols+vvValues
    \param[in] A is general nonsquare, nonsymmetric matrix
    \param[in] B is square symmetric matrix, containing only lower triangular part
    \param[in] alpha, beta are constants
  
  */
  void transAAplusB(hiopMatrixSparse &C, vector<vector<int>> &vvCols_C, vector<vector<double>> &vvValues_C, bool structureNotInitialized, double alpha, const hiopMatrixSparse &A, double beta, const hiopMatrixSparse &B);

  void transAAplusB2(hiopMatrixSparse &C, vector<vector<int>> &vvCols_C, vector<vector<double>> &vvValues_C, bool structureNotInitialized, double alpha, const hiopMatrixSparse &A, double beta, const hiopMatrixSparse &B);

  /** 
   Append scaled jacobian and identity (blocks 2-1 and 2-2)
   to the sparse matrix H containing only H_xx block
    
   H_xx = _hessianAugLagr
   H_sx = -2*rho*Jineq'
   H_ss = 2*rho*I
       | Hxx   0  |
   H = |          |
       | Hsx  Hss |

   The method can work with #H being either an empty sparse matrix,
    i.e. hiopMatrixSparse(0,0,0.), in which case the storage is allocated
    and the sparse structure is created using vvCols and vvValues.
    In case #H already contains all the required storage space and sparse structure,
    we only update the numerical values of the nonzeros.
    
    \param[out] H 
    \param[out] vvCols, vvValues The method computes and returns only the lower triangular part of the symmetric result.
    \param[in] structureNotInitialized Switch deciding which output will be updated, either H or vvCols+vvValues
    \param[in] J is general nonsquare, nonsymmetric matrix
    \param[in] alpha
    \param[in] cons_ineq_mapping Specifies rows of J corresponding to Jineq
  */
  void appendScaledJacobian(hiopMatrixSparse &H, vector<vector<int>> &vvCols_H, vector<vector<double>> &vvValues_H, bool structureNotInitialized, double alpha, const hiopMatrixSparse &J, long long *cons_ineq_mapping);

#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  //bool assertSymmetry(double tol=1e-16) const;
#endif

private:
    NLP_CLASS_IN     *nlp_in; ///< input NLP representation, neede to evaluate NLP hessian
    int nvars_nlp;  ///<property of the input NLP
    int nslacks_nlp;  ///<property of the input NLP
    int mcons_nlp;  ///<property of the input NLP
    int nnz_nlp;   ///<property of the input NLP
    hiopVectorPar    *_lambdaForHessEval; ///< lambda + 2*rho*c(x), used during the NLP Hessian evaluation
    hiopMatrixSparse *_hessianNlp; ///< Hessian of Lagrangian of the original NLP problem evaluated with extended #_lambdaForHessEval
    hiopMatrixSparse *_hessianAugLagr; ///< Hessian of the AL problem

    int _updateIterator; ///<iterator used when updating directly nonzeros in the #_hessianAugLagr
};


}
#endif
