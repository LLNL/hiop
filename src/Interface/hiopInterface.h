// The C interface header used by the user. This needs a detailed user documentation.

//include hiop index and size types
#include "hiop_types.h"

typedef struct cHiopProblem {
  void *refcppHiop; // Pointer to the cpp object
  void *hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void *user_data; 
  double *solution;
  double obj_value;
  int (*get_starting_point)(hiop_size_type n_, double* x0, void* jprob); 
  int (*get_prob_sizes)(hiop_size_type* n_, hiop_size_type* m_, void* jprob); 
  int (*get_vars_info)(hiop_size_type n, double *xlow_, double* xupp_, void* jprob);
  int (*get_cons_info)(hiop_size_type m, double *clow_, double* cupp_, void* jprob);
  int (*eval_f)(hiop_size_type n, double* x, int new_x, double* obj, void* jprob);
  int (*eval_grad_f)(hiop_size_type n, double* x, int new_x, double* gradf, void* jprob);
  int (*eval_cons)(hiop_size_type n, hiop_size_type m,
    double* x, int new_x, 
    double* cons, void* jprob);
  int (*get_sparse_dense_blocks_info)(hiop_size_type* nx_sparse, hiop_size_type* nx_dense,
    hiop_size_type* nnz_sparse_Jaceq, hiop_size_type* nnz_sparse_Jacineq,
    hiop_size_type* nnz_sparse_Hess_Lagr_SS, 
    hiop_size_type* nnz_sparse_Hess_Lagr_SD, void* jprob);
  int (*eval_Jac_cons)(hiop_size_type n, hiop_size_type m,
    double* x, int new_x,
    hiop_size_type nsparse, hiop_size_type ndense, 
    hiop_size_type nnzJacS, hiop_index_type* iJacS, hiop_index_type* jJacS, double* MJacS, 
    double* JacD, void *jprob);
  int (*eval_Hess_Lagr)(hiop_size_type n, hiop_size_type m,
    double* x, int new_x, double obj_factor,
    double* lambda, int new_lambda,
    hiop_size_type nsparse, hiop_size_type ndense, 
    hiop_size_type nnzHSS, hiop_index_type* iHSS, hiop_index_type* jHSS, double* MHSS, 
    double* HDD,
    hiop_size_type nnzHSD, hiop_index_type* iHSD, hiop_index_type* jHSD, double* MHSD, void* jprob);
} cHiopProblem;
extern int hiop_createProblem(cHiopProblem *problem);
extern int hiop_solveProblem(cHiopProblem *problem);
extern int hiop_destroyProblem(cHiopProblem *problem);
