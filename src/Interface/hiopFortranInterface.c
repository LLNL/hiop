// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

/**
 * @file hiopFortranInterface.c
 *
 * @author Michel Schanen <mschanen@anl.gov>, ANL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LNNL
 *
 */

// The C interface header used by the user. This needs a detailed user documentation.

//include hiop index and size types
#include "hiop_types.h"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "hiopInterface.h"
#include "hiopFortranInterface.h"
//#include "hiopAlgFilterIPM.hpp"

int get_starting_point_wrapper( hiop_size_type* n_, double* x0, void* user_data)
{
   FProb* fprob = (FProb*) user_data;
  for(int i=0; i<fprob->n; i=i+1) {
    x0[i] = fprob->x0[i];
  }
   return 0;
}

int get_prob_sizes_wrapper( hiop_size_type* n_, hiop_size_type* m_, void* user_data)
{
   FProb* fprob = (FProb*) user_data;
   *n_ = fprob->n;
   *m_ = fprob->m;
   return 0;
}

int get_vars_info_wrapper(hiop_size_type n, double *xlow_, double* xupp_, void* user_data) 
{
  FProb* fprob = (FProb*) user_data;
  hiop_size_type i = 0;
  for(i=0; i<fprob->n; i=i+1) {
    xlow_[i] = fprob->xlow[i];
    xupp_[i] = fprob->xupp[i];
  }
  return 0;
}

int get_cons_info_wrapper(hiop_size_type m, double *clow_, double* cupp_, void* user_data) 
{
  FProb* fprob = (FProb*) user_data;
  hiop_size_type i = 0;
  for(i=0; i<fprob->m; i=i+1) {
    clow_[i] = fprob->clow[i];
    cupp_[i] = fprob->cupp[i];
  }
  return 0;
}

int get_sparse_blocks_info_wrapper(hiop_size_type* nx,
                           hiop_size_type* nnz_sparse_Jaceq,
                           hiop_size_type* nnz_sparse_Jacineq,
                           hiop_size_type* nnz_sparse_Hess_Lagr,
                           void* user_data) 
{
  FProb* fprob = (FProb*) user_data;
  *nx = fprob->n;
  *nnz_sparse_Jaceq = fprob->nnz_sparse_Jaceq;
  *nnz_sparse_Jacineq = fprob->nnz_sparse_Jacineq;
  *nnz_sparse_Hess_Lagr = fprob->nnz_sparse_Hess_Lagr;
  return 0;
}

int eval_f_wrapper(hiop_size_type n, double* x, int new_x, double* obj, void* user_data) 
{
  hiop_size_type n_ = n;
  hiop_size_type new_x_ = new_x;
  FProb* fprob = (FProb*) user_data;
  fprob->f_eval_f(&n_, x, &new_x_, obj);
  return 0;
}

int eval_c_wrapper(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* user_data) 
{
  hiop_size_type m_ = m;
  hiop_size_type new_x_ = new_x;
  FProb* fprob = (FProb*) user_data;
  hiop_size_type n_ = fprob->n;
  fprob->f_eval_c(&n_, &m_, x, &new_x_, cons);
  return 0;
}

int eval_grad_wrapper(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data) 
{
  hiop_size_type n_ = n;
  hiop_size_type new_x_ = new_x;
  FProb* fprob = (FProb*) user_data;
  fprob->f_eval_grad(&n_, x, &new_x_, gradf);
  return 0;
}

int eval_jac_wrapper(hiop_size_type n,
                     hiop_size_type m,
                     double* x,
                     int new_x,
                     int nnzJacS,
                     hiop_index_type* iJacS,
                     hiop_index_type* jJacS,
                     double* MJacS,
                     void *user_data) 
{ 
  hiop_size_type n_ = n;
  hiop_size_type m_ = m;
  hiop_size_type new_x_ = new_x;
  hiop_size_type nnzJacS_ = nnzJacS;
  int task = 0;
  FProb* fprob = (FProb*) user_data;
  if(iJacS==NULL && jJacS==NULL && MJacS !=NULL) {
    task = 1;
  } else if(iJacS!=NULL && jJacS!=NULL && MJacS ==NULL){
    task = 0;
  } else {
    printf("ERROR: cannot reach here!");
  }
  fprob->f_eval_jac(&task, &n_, &m_, x, &new_x_, &nnzJacS_, iJacS, jJacS, MJacS);

  // TODO: If we can add a flag in ma57 when a problem comes from fortran?
  //       Then we don't need to -1 here and +1 when initialize ma57
  if(task == 0) {
    for(hiop_index_type k=0; k<nnzJacS_; k++){
      iJacS[k] -= 1;
      jJacS[k] -= 1;
    }
  }
  return 0;
}

int eval_hess_wrapper(hiop_size_type n,
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
                          void* user_data) 
{
  hiop_size_type n_ = n;
  hiop_size_type m_ = m;
  hiop_size_type new_x_ = new_x;
  hiop_size_type new_lambda_ = new_lambda;
  hiop_size_type nnzHSS_ = nnzHSS;
  double obj_scal_ = obj_factor;
  int task = 0;

  FProb* fprob = (FProb*) user_data;
  if(iHSS==NULL && jHSS==NULL && MHSS !=NULL) {
    task = 1;
  } else if(iHSS!=NULL && jHSS!=NULL && MHSS ==NULL){
    task = 0;
  } else {
    printf("ERROR: cannot reach here!");
  }
  fprob->f_eval_hess(&task, &n_, &m_, &obj_scal_, x, &new_x_, lambda, &new_lambda_, &nnzHSS_, iHSS, jHSS, MHSS);

  // TODO: If we can add a flag in ma57 when a problem comes from fortran?
  //       Then we don't need to -1 here and +1 when initialize ma57
  if(task == 0) {
    for(hiop_index_type k=0; k<nnzHSS_; k++){
      iHSS[k] -= 1;
      jHSS[k] -= 1;
    }
  }
  return 0;
}

void* FC_GLOBAL(hiopsparseprob, HIOPSPARSEPROB) ( hiop_size_type*   n,
                                  hiop_size_type*   m,
                                  hiop_size_type*   nnz_sparse_Jaceq,
                                  hiop_size_type*   nnz_sparse_Jacineq,
                                  hiop_size_type*   nnz_sparse_Hess_Lagr,
                                  double*           xlow,
                                  double*           xupp,
                                  double*           clow,
                                  double*           cupp,
                                  double*           x0,
                                  f_eval_f_cb       f_eval_f,
                                  f_eval_c_cb       f_eval_c,
                                  f_eval_grad_cb    f_eval_grad,
                                  f_eval_jac_cb     f_eval_jac,
                                  f_eval_hess_cb    f_eval_hess)
{
  FProb* f_prob = (FProb*) malloc(sizeof(FProb));
  f_prob->n = *n;
  f_prob->m = *m;
  f_prob->nnz_sparse_Jaceq = *nnz_sparse_Jaceq;
  f_prob->nnz_sparse_Jacineq = *nnz_sparse_Jacineq;
  f_prob->nnz_sparse_Hess_Lagr = *nnz_sparse_Hess_Lagr;
  
  f_prob->c_prob  = (cHiopSparseProblem*) malloc(sizeof(cHiopSparseProblem));
  
  if( f_prob->c_prob == NULL )
  {
    free(f_prob);
    return (void*) NULL;
  }

  f_prob->xlow = (double*) malloc(f_prob->n*sizeof(double));
  f_prob->xupp = (double*) malloc(f_prob->n*sizeof(double));
  for(int i=0; i<f_prob->n; i++) {
    f_prob->xlow[i] = xlow[i];
    f_prob->xupp[i] = xupp[i];
  }
  f_prob->clow = (double*) malloc(f_prob->m*sizeof(double));
  f_prob->cupp = (double*) malloc(f_prob->m*sizeof(double));
  for(int i=0; i<f_prob->m; i++) {
    f_prob->clow[i] = clow[i];
    f_prob->cupp[i] = cupp[i];
  }

  f_prob->x0 = (double*) malloc(f_prob->n*sizeof(double));
  for(int i=0; i<f_prob->n; i++) {
    f_prob->x0[i] = x0[i];
  }

  f_prob->f_eval_f = f_eval_f;
  f_prob->f_eval_c = f_eval_c;
  f_prob->f_eval_grad = f_eval_grad;
  f_prob->f_eval_jac = f_eval_jac;
  f_prob->f_eval_hess = f_eval_hess;
  
  f_prob->c_prob->user_data = f_prob;

  f_prob->c_prob->get_prob_sizes = get_prob_sizes_wrapper;  
  f_prob->c_prob->get_vars_info = get_vars_info_wrapper;
  f_prob->c_prob->get_cons_info = get_cons_info_wrapper;
  f_prob->c_prob->get_starting_point = get_starting_point_wrapper;
  f_prob->c_prob->get_sparse_blocks_info = get_sparse_blocks_info_wrapper;
  f_prob->c_prob->eval_f = eval_f_wrapper;
  f_prob->c_prob->eval_cons = eval_c_wrapper;
  f_prob->c_prob->eval_grad_f = eval_grad_wrapper;
  f_prob->c_prob->eval_Jac_cons = eval_jac_wrapper;
  f_prob->c_prob->eval_Hess_Lagr = eval_hess_wrapper;

  f_prob->c_prob->solution = (double*)malloc(f_prob->n * sizeof(double));
  for(int i=0; i<f_prob->n; i++) {
    f_prob->c_prob->solution[i] = 0.0;
  }

  hiop_sparse_create_problem(f_prob->c_prob);

  printf("Creat HIOP_SPARSE problem from Fortran interface!\n");
  return (void*) f_prob;
}

void FC_GLOBAL(hiopsparsesolve, HIOPSPARSESOLVE) (void** f_prob_in, double* obj, double* sol)
{
  FProb* f_prob = (FProb*) *f_prob_in;
  cHiopSparseProblem *prob = f_prob->c_prob;

  hiop_sparse_solve_problem(prob);

  for(int i=0; i<f_prob->n; i++) {
    sol[i] = prob->solution[i];
  }
  
  *obj = prob->obj_value;
  printf("Solve HIOP_SPARSE problem from Fortran interface!\n");
}

void FC_GLOBAL(deletehiopsparseprob, DELETEHIOPSPARSEPROB) (void** f_prob_in)
{
  FProb* f_prob = (FProb*) *f_prob_in;
  hiop_sparse_destroy_problem(f_prob->c_prob);

  free(f_prob->xlow);
  free(f_prob->xupp);
  free(f_prob->clow);
  free(f_prob->cupp);
  free(f_prob->c_prob->solution);
  free(f_prob->c_prob);
  free(f_prob);
  *f_prob_in = (void*)NULL;
  printf("Delete HIOP_SPARSE problem from Fortran interface!\n");
}