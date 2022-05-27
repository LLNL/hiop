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
#include <stdlib.h>
#include <stdio.h>
#include "hiopFortranInterface.h"
#include "FortranCInterface.hpp"

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

int get_sparse_blocks_info(hiop_size_type* nx,
                           hiop_size_type* nnz_sparse_Jaceq,
                           hiop_size_type* nnz_sparse_Jacineq,
                           hiop_size_type* nnz_sparse_Hess_Lagr,
                           void* user_data) 
{
  FProb* fprob = (FProb*) user_data;
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
  return 0;
}

int eval_grad_wrapper(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data) 
{
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
  f_prob->clow = (double*) malloc(f_prob->m*sizeof(double));
  f_prob->cupp = (double*) malloc(f_prob->m*sizeof(double));
  f_prob->f_eval_f = f_eval_f;
  f_prob->f_eval_c = f_eval_c;
  f_prob->f_eval_grad = f_eval_grad;
  f_prob->f_eval_jac = f_eval_jac;
  f_prob->f_eval_hess = f_eval_hess;
  
  f_prob->c_prob->user_data = f_prob;

  f_prob->c_prob->get_prob_sizes = get_prob_sizes_wrapper;  
  f_prob->c_prob->get_vars_info = get_vars_info_wrapper;
  f_prob->c_prob->get_cons_info = get_cons_info_wrapper;
  f_prob->c_prob->eval_f = eval_f_wrapper;
  f_prob->c_prob->eval_cons = eval_c_wrapper;
  f_prob->c_prob->eval_grad_f = eval_grad_wrapper;
  f_prob->c_prob->eval_Jac_cons = eval_jac_wrapper;
  f_prob->c_prob->eval_Hess_Lagr = eval_hess_wrapper;



#if 0
  c_prob.eval_f = eval_f;
  c_prob.eval_grad_f = eval_grad_f;
  c_prob.eval_cons = eval_cons;
  c_prob.get_sparse_blocks_info = get_sparse_blocks_info;
  c_prob.eval_Jac_cons = eval_Jac_cons;
  c_prob.eval_Hess_Lagr = eval_Hess_Lagr;
  c_prob.get_starting_point = get_starting_point_wrapper;

  c_prob.solution = (double*)malloc(n * sizeof(double));
  c_prob.solution = (double*)malloc(1* sizeof(double));

   /* First create a new IpoptProblem object; if that fails return 0 */
   f_prob->c_prob = CreateIpoptProblem(n, X_L, X_U, m, G_L, G_U, nele_jac, nele_hess,
                         index_style, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
   if( fuser_data->Problem == NULL )
   {
      free(fuser_data);
      return (fptr)NULL;
   }

   /* Store the information for the callback function */
   fuser_data->eval_f = eval_f;
   fuser_data->eval_c = eval_c;
   fuser_data->eva = EVAL_GRAD_F;
   fuser_data->EVAL_JAC_G = EVAL_JAC_G;
   fuser_data->EVAL_HESS = EVAL_HESS;
   fuser_data->INTERMEDIATE_CB = NULL;
#endif
    printf("Creat HIOP_SPARSE problem from Fortran interface!\n");
    return (void*) f_prob;
}

void FC_GLOBAL(deletehiopsparseprob, DELETEHIOPSPARSEPROB) (void** f_prob_in)
{
  FProb* f_prob = (FProb*) *f_prob_in;

  free(f_prob->xlow);
  free(f_prob->xupp);
  free(f_prob->clow);
  free(f_prob->cupp);
  free(f_prob->c_prob);
  free(f_prob);
  *f_prob_in = (void*)NULL;
  printf("Delete HIOP_SPARSE problem from Fortran interface!\n");
}