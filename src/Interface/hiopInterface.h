// The C interface header used by the user. This needs a detailed user documentation.

/** The hiop_int_type type defined below needs to match the C++ type hiop::int_type defined for 
 * HiOp in the C++ header file hiop_defs.hpp
 */
typedef int hiop_int_type;


typedef struct cHiopProblem {
  void *refcppHiop; // Pointer to the cpp object
  void *hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void *user_data; 
  double *solution;
  double obj_value;
  int (*get_starting_point)(hiop_int_type n_, double* x0, void* jprob); 
  int (*get_prob_sizes)(hiop_int_type* n_, hiop_int_type* m_, void* jprob); 
  int (*get_vars_info)(hiop_int_type n, double *xlow_, double* xupp_, void* jprob);
  int (*get_cons_info)(hiop_int_type m, double *clow_, double* cupp_, void* jprob);
  int (*eval_f)(int n, double* x, int new_x, double* obj, void* jprob);
  int (*eval_grad_f)(hiop_int_type n, double* x, int new_x, double* gradf, void* jprob);
  int (*eval_cons)(hiop_int_type n, hiop_int_type m,
    double* x, int new_x, 
    double* cons, void* jprob);
  int (*get_sparse_dense_blocks_info)(int* nx_sparse, int* nx_dense,
    int* nnz_sparse_Jaceq, int* nnz_sparse_Jacineq,
    int* nnz_sparse_Hess_Lagr_SS, 
    int* nnz_sparse_Hess_Lagr_SD, void* jprob);
  int (*eval_Jac_cons)(hiop_int_type n, hiop_int_type m,
    double* x, int new_x,
    hiop_int_type nsparse, hiop_int_type ndense, 
    int nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
    double* JacD, void *jprob);
  int (*eval_Hess_Lagr)(hiop_int_type n, hiop_int_type m,
    double* x, int new_x, double obj_factor,
    double* lambda, int new_lambda,
    hiop_int_type nsparse, hiop_int_type ndense, 
    int nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
    double* HDD,
    int nnzHSD, int* iHSD, int* jHSD, double* MHSD, void* jprob);
} cHiopProblem;
extern int hiop_createProblem(cHiopProblem *problem);
extern int hiop_solveProblem(cHiopProblem *problem);
extern int hiop_destroyProblem(cHiopProblem *problem);
