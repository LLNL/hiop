#ifndef CHIOP_INTERFACE_HPP
#define CHIOP_INTERFACE_HPP
#include "hiop_defs.hpp"
#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

/** Light C interface that wraps around the mixed-dense nlp class in HiOp. Its initial motivation
 * was to serve as an interface to Julia
 */

using namespace hiop;
class cppUserProblem;
extern "C" {
  // C struct with HiOp function callbacks
  typedef struct cHiopProblem {
    hiopNlpMDS *refcppHiop;
    cppUserProblem *hiopinterface;
    // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
    void *user_data;
    // Used by hiop_solveProblem() to store the final state. The duals should be added here.
    double *solution;
    double obj_value;
    // HiOp callback function wrappers
    int (*get_starting_point)(size_type n_, double* x0, void* user_data); 
    int (*get_prob_sizes)(size_type* n_, size_type* m_, void* user_data); 
    int (*get_vars_info)(size_type n, double *xlow_, double* xupp_, void* user_data);
    int (*get_cons_info)(size_type m, double *clow_, double* cupp_, void* user_data);
    int (*eval_f)(size_type n, double* x, int new_x, double* obj, void* user_data);
    int (*eval_grad_f)(size_type n, double* x, int new_x, double* gradf, void* user_data);
    int (*eval_cons)(size_type n, size_type m,
      double* x, int new_x, 
      double* cons, void* user_data);
    int (*get_sparse_dense_blocks_info)(hiop_size_type* nx_sparse, hiop_size_type* nx_dense,
      hiop_size_type* nnz_sparse_Jaceq, hiop_size_type* nnz_sparse_Jacineq,
      hiop_size_type* nnz_sparse_Hess_Lagr_SS, 
      hiop_size_type* nnz_sparse_Hess_Lagr_SD, void* user_data);
    int (*eval_Jac_cons)(size_type n, size_type m,
      double* x, int new_x,
      size_type nsparse, size_type ndense, 
      int nnzJacS, hiop_index_type* iJacS, hiop_index_type* jJacS, double* MJacS, 
      double* JacD, void *user_data);
    int (*eval_Hess_Lagr)(size_type n, size_type m,
      double* x, int new_x, double obj_factor,
      double* lambda, int new_lambda,
      size_type nsparse, size_type ndense, 
      hiop_size_type nnzHSS, hiop_index_type* iHSS, hiop_index_type* jHSS, double* MHSS, 
      double* HDD,
      hiop_size_type nnzHSD, hiop_index_type* iHSD, hiop_index_type* jHSD, double* MHSD, void* user_data);
  } cHiopProblem;
}


// The cpp object used in the C interface
class cppUserProblem : public hiopInterfaceMDS
{
  public:
    cppUserProblem(cHiopProblem *cprob_)
      : cprob(cprob_) 
    {
    }

    virtual ~cppUserProblem()
    {
    }
    // HiOp callbacks calling the C wrappers
    bool get_prob_sizes(size_type& n_, size_type& m_) 
    {
      cprob->get_prob_sizes(&n_, &m_, cprob->user_data);
      return true;
    };
    bool get_starting_point(const size_type& n, double *x0)
    {
      cprob->get_starting_point(n, x0, cprob->user_data);
      return true;
    };
    bool get_vars_info(const size_type& n, double *xlow_, double* xupp_, NonlinearityType* type)
    {
      for(size_type i=0; i<n; ++i) type[i]=hiopNonlinear;
      cprob->get_vars_info(n, xlow_, xupp_, cprob->user_data);
      return true;
    };
    bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
    {
      for(size_type i=0; i<m; ++i) type[i]=hiopNonlinear;
      cprob->get_cons_info(m, clow, cupp, cprob->user_data);
      return true;
    };
    bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
    {
      cprob->eval_f(n, (double *) x, 0, &obj_value, cprob->user_data);
      return true;
    };

    bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
    {
      cprob->eval_grad_f(n, (double *) x, 0, gradf, cprob->user_data);

      return true;
    };
    bool eval_cons(const size_type& n, const size_type& m,
      const size_type& num_cons, const hiop_index_type* idx_cons,  
      const double* x, bool new_x, 
      double* cons)
    {
      return false;
    };
    bool eval_cons(const size_type& n, const size_type& m, 
      const double* x, bool new_x, double* cons)
    {
      cprob->eval_cons(n, m, (double *) x, new_x, cons, cprob->user_data);
      return true;
    };
    bool get_sparse_dense_blocks_info(hiop_size_type& nx_sparse, hiop_size_type& nx_dense,
      hiop_size_type& nnz_sparse_Jaceq, hiop_size_type& nnz_sparse_Jacineq,
      hiop_size_type& nnz_sparse_Hess_Lagr_SS, 
      hiop_size_type& nnz_sparse_Hess_Lagr_SD)
    {
      cprob->get_sparse_dense_blocks_info(&nx_sparse, &nx_dense, &nnz_sparse_Jaceq, &nnz_sparse_Jacineq, 
                                          &nnz_sparse_Hess_Lagr_SS, &nnz_sparse_Hess_Lagr_SD, cprob->user_data);
      return true;
    };
    bool eval_Jac_cons(const size_type& n, const size_type& m,
      const size_type& num_cons, const hiop_index_type* idx_cons,
      const double* x, bool new_x,
      const size_type& nsparse, const size_type& ndense, 
      const hiop_size_type& nnzJacS, hiop_index_type* iJacS, hiop_index_type* jJacS, double* MJacS, 
      double* JacD)
    {
      return false;
    };
    bool eval_Jac_cons(const size_type& n, const size_type& m,
      const double* x, bool new_x,
      const size_type& nsparse, const size_type& ndense, 
      const hiop_size_type& nnzJacS, hiop_index_type* iJacS, hiop_index_type* jJacS, double* MJacS, 
      double* JacD)
    {
      cprob->eval_Jac_cons(n, m, (double *) x, new_x, nsparse, ndense, 
                           nnzJacS, iJacS, jJacS, MJacS,
                           JacD, cprob->user_data);
      return true;
    };
    bool eval_Hess_Lagr(const size_type& n, const size_type& m,
      const double* x, bool new_x, const double& obj_factor,
      const double* lambda, bool new_lambda,
      const size_type& nsparse, const size_type& ndense, 
      const hiop_size_type& nnzHSS, hiop_index_type* iHSS, hiop_index_type* jHSS, double* MHSS, 
      double* HDD,
      hiop_size_type& nnzHSD, hiop_index_type* iHSD, hiop_index_type* jHSD, double* MHSD)
    {
      //Note: lambda is not used since all the constraints are linear and, therefore, do 
      //not contribute to the Hessian of the Lagrangian
      cprob->eval_Hess_Lagr(n, m, (double *) x, new_x, obj_factor,
                            (double *) lambda, new_lambda, nsparse, ndense,
                            nnzHSS, iHSS, jHSS, MHSS, 
                            HDD, 
                            nnzHSD, iHSD, jHSD, MHSD,
                            cprob->user_data);
      return true;
    };
private:
  // Storing the C struct in the CPP object
  cHiopProblem *cprob;
};

/** The 3 essential function calls to create and destroy a problem object in addition to solve a problem.
 * Some option setters will be added in the future.
 */
extern "C" int hiop_createProblem(cHiopProblem *problem);
extern "C" int hiop_solveProblem(cHiopProblem *problem);
extern "C" int hiop_destroyProblem(cHiopProblem *problem);
#endif
