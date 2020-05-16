// The C interface header used by the user. This needs a detailed user documentation.

typedef struct cHiopProblem {
  void *refcppHiop; // Pointer to the cpp object
  void *hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void *user_data; 
  double *solution;
  double obj_value;
  int (*get_starting_point)(long long n_, double* x0, void* jprob); 
  int (*get_prob_sizes)(long long* n_, long long* m_, void* jprob); 
  int (*get_vars_info)(long long n, double *xlow_, double* xupp_, void* jprob);
  int (*get_cons_info)(long long m, double *clow_, double* cupp_, void* jprob);
  int (*eval_f)(int n, double* x, int new_x, double* obj, void* jprob);
  int (*eval_grad_f)(long long n, double* x, int new_x, double* gradf, void* jprob);
  int (*eval_cons)(long long n, long long m,
    double* x, int new_x, 
    double* cons, void* jprob);
  int (*get_sparse_dense_blocks_info)(int* nx_sparse, int* nx_dense,
    int* nnz_sparse_Jaceq, int* nnz_sparse_Jacineq,
    int* nnz_sparse_Hess_Lagr_SS, 
    int* nnz_sparse_Hess_Lagr_SD, void* jprob);
  int (*eval_Jac_cons)(long long n, long long m,
    double* x, int new_x,
    long long nsparse, long long ndense, 
    int nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
    double* JacD, void *jprob);
  int (*eval_Hess_Lagr)(long long n, long long m,
    double* x, int new_x, double obj_factor,
    double* lambda, int new_lambda,
    long long nsparse, long long ndense, 
    int nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
    double* HDD,
    int nnzHSD, int* iHSD, int* jHSD, double* MHSD, void* jprob);
} cHiopProblem;
extern int hiop_createProblem(cHiopProblem *problem);
extern int hiop_solveProblem(cHiopProblem *problem);
extern int hiop_destroyProblem(cHiopProblem *problem);