#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "hiopInterface.h"
#include <math.h>

typedef struct settings {
  hiop_size_type n; hiop_size_type m;
  hiop_size_type nnz_sparse_Jaceq; hiop_size_type nnz_sparse_Jacineq;
  hiop_size_type nnz_sparse_Hess_Lagr;
  double* xlow; double* xupp; double* clow; double* cupp;
} settings;


int get_prob_sizes(hiop_size_type* n_, hiop_size_type* m_, void* user_data_) 
{
  settings* user_data = (settings*) user_data_;
  *n_ = user_data->n;
  *m_ = user_data->m;
  return 0;
} 

int get_vars_info(hiop_size_type n, double *xlow_, double* xupp_, void* user_data_) 
{
  settings* user_data = (settings*) user_data_;
  hiop_size_type i = 0;
  for(i=0; i<user_data->n; i=i+1) {
    xlow_[i] = user_data->xlow[i];
    xupp_[i] = user_data->xupp[i];
  }
  return 0;
}

int get_cons_info(hiop_size_type m, double *clow_, double* cupp_, void* user_data_) 
{
  settings* user_data = (settings*) user_data_;
  hiop_size_type i = 0;
  for(i=0; i<user_data->m; i=i+1) {
    clow_[i] = user_data->clow[i];
    cupp_[i] = user_data->cupp[i];
  }
  return 0;
}

int get_sparse_blocks_info(hiop_size_type* nx,
                           hiop_size_type* nnz_sparse_Jaceq,
                           hiop_size_type* nnz_sparse_Jacineq,
                           hiop_size_type* nnz_sparse_Hess_Lagr,
                           void* user_data_) {
  settings* user_data = (settings*) user_data_;
  *nx = user_data->n;
  *nnz_sparse_Jaceq = user_data->nnz_sparse_Jaceq;
  *nnz_sparse_Jacineq = user_data->nnz_sparse_Jacineq;
  *nnz_sparse_Hess_Lagr = user_data->nnz_sparse_Hess_Lagr;
  return 0;
}

int eval_f(hiop_size_type n, double* x, int new_x, double* obj, void* user_data_) 
{
  settings* user_data = (settings*) user_data_;
  int i = 0;
  *obj = 0.;
  for(i=0; i<user_data->n; i=i+1) {
    *obj += 0.25*(x[i]-1.)*(x[i]-1.)*(x[i]-1.)*(x[i]-1.);
  }

  return 0;
}

int eval_grad_f(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data_)
{
  settings* user_data = (settings*) user_data_;
  int i = 0;
  for(i=0; i<user_data->n; i=i+1) {
    gradf[i] = (x[i]-1.)*(x[i]-1.)*(x[i]-1.);
  }

  return 0;
}

int eval_cons(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* user_data_) 
{
  settings* user_data = (settings*) user_data_;
  assert( 2+n-3 == user_data->m);

  int con_idx = 0;
  for(con_idx = 0; con_idx < user_data->m; ++con_idx) {
    cons[con_idx]=0.;
  }
  con_idx = 0;

  //compute the constraint one by one.
  // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
  cons[con_idx++] += ( 4*x[0] + 2*x[1]);

  // --- constraint 2 body ---> 2*x_1 + x_3
  cons[con_idx++] += ( 2*x[0] + 1*x[2]);

  // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
  for(int i = 3; i < n; i++) {
      cons[con_idx++] += ( 2*x[0] + 0.5*x[i]);
  }

  return 0;
}

int eval_Jac_cons(hiop_size_type n,
                  hiop_size_type m,
                  double* x,
                  int new_x,
                  hiop_size_type nnzJacS,
                  hiop_index_type* iJacS,
                  hiop_index_type* jJacS,
                  double* MJacS, 
                  void* user_data_)
{
  settings* user_data = (settings*) user_data_;
  assert(m==user_data->m);
  assert(nnzJacS == 4 + 2*(n-3));

  int nnzit = 0;
  int conidx = 0;

  if(iJacS!=NULL && jJacS!=NULL){
    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 1;
    conidx++;

    // --- constraint 2 body ---> 2*x_1 + x_3
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 2;
    conidx++;

    // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
    for(int i=3; i<n; i++){
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = i;
      conidx++;
    }
    assert(nnzit == nnzJacS);
  }

  //values for sparse Jacobian if requested by the solver
  nnzit = 0;
  if(MJacS!=NULL) {
    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    MJacS[nnzit++] = 4;
    MJacS[nnzit++] = 2;

    // --- constraint 2 body ---> 2*x_1 + x_3
    MJacS[nnzit++] = 2;
    MJacS[nnzit++] = 1;

    // --- constraint 3 body --->   2*x_1 + 0.5*x_4
    for(int i=3; i<n; i++){
      MJacS[nnzit++] = 2;
      MJacS[nnzit++] = 0.5;
    }
    assert(nnzit == nnzJacS);
  }
  return 0;
}

int eval_Hess_Lagr(hiop_size_type n,
                   hiop_size_type m,
                   double* x,
                   int new_x,
                   double obj_factor,
                   double* lambda,
                   int new_lambda,
                   hiop_size_type nnzHSS,
                   hiop_index_type* iHSS,
                   hiop_index_type* jHSS,
                   double* MHSS,
                   void* user_data_)
{
  settings* user_data = (settings*) user_data_;
  //Note: lambda is not used since all the constraints are linear and, therefore, do 
  //not contribute to the Hessian of the Lagrangian

  assert(nnzHSS==user_data->n);
 
  if(iHSS!=NULL && jHSS!=NULL) {
    for(int i=0; i<n; i++) {
      iHSS[i] = i;
      jHSS[i] = i;
    }
  }

  if(MHSS!=NULL) {
    for(int i=0; i<n; i++) {
      MHSS[i] = obj_factor * 3*(x[i]-1.)*(x[i]-1.);
    }
  }
  return 0;
}

int get_starting_point(const hiop_size_type n, double* x0, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  int i = 0;
  for(i=0; i<user_data->n; i=i+1) {
    x0[i] = 0.0;
  }
  return 0;
}

int main(int argc, char **argv) {
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size); assert(MPI_SUCCESS==ierr);
  //int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf("[error] driver detected more than one rank but the driver should be run "
           "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

  hiop_size_type n = 500;
  hiop_size_type m = 499;

  hiop_size_type nnz_sparse_Jaceq = 2;
  hiop_size_type nnz_sparse_Jacineq = 2 + 2*(n-3);
  hiop_size_type nnz_sparse_Hess_Lagr = n;
  
  double* xlow = (double*)malloc(n*sizeof(double));
  double* xupp = (double*)malloc(n*sizeof(double));
  for(int i=0; i<n; i++) {
    if(i==0) { 
      xlow[i] = -1e20;
      xupp[i] = 1e20;
      continue;
    }
    if(i==1) {
      xlow[i] = 0.0;
      xupp[i] = 1e20;
      continue;
    }
    if(i==2) {
      xlow[i] = 1.5;
      xupp[i] = 10.0;
      continue;
    }
    //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
    xlow[i] = 0.5;
    xupp[i] = 1e20;
  }

  double* clow = (double*)malloc(m*sizeof(double));
  double* cupp = (double*)malloc(m*sizeof(double));
  int conidx = 0;
  clow[conidx] = 10.0;
  cupp[conidx++] = 10.0;
  clow[conidx] = 5.0;
  cupp[conidx++] = 1e20;
  for(int i=3; i<n; i++) {
    clow[conidx] = 1.0;
    cupp[conidx++] = 2*n;
  }

  settings user_data = {n, m, nnz_sparse_Jaceq, nnz_sparse_Jacineq,
                        nnz_sparse_Hess_Lagr,
                        xlow, xupp, clow, cupp};
                        
  cHiopSparseProblem problem;
  problem.user_data_ = &user_data;
  problem.get_starting_point_ = get_starting_point;
  problem.get_prob_sizes_ = get_prob_sizes;  
  problem.get_vars_info_ = get_vars_info;
  problem.get_cons_info_ = get_cons_info;
  problem.eval_f_ = eval_f;
  problem.eval_grad_f_ = eval_grad_f;
  problem.eval_cons_ = eval_cons;
  problem.get_sparse_blocks_info_ = get_sparse_blocks_info;
  problem.eval_Jac_cons_ = eval_Jac_cons;
  problem.eval_Hess_Lagr_ = eval_Hess_Lagr;
  problem.solution_ = (double*)malloc(n * sizeof(double));
  for(int i=0; i<n; i++) {
    problem.solution_[i] = 0.0;
  }

  hiop_sparse_create_problem(&problem);
  hiop_sparse_solve_problem(&problem);
  if(fabs(problem.obj_value_-(1.10351566513480e-01))>1e-6) {
    printf("objective mismatch for Sparse Ex1 C interface problem with 500 "
      "variables did. BTW, obj=%18.12e was returned by HiOp.\n", problem.obj_value_);
      return -1;
  }
  
  printf("Optimal objective: %22.14e. Solver status: %d. Number of iterations: %d\n", 
             problem.obj_value_, problem.status_, problem.niters_);

  hiop_sparse_destroy_problem(&problem);
  free(problem.solution_);
  free(xlow);
  free(xupp);
  free(clow);
  free(cupp);

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
