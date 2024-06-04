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
 * @file hiopFortranInterface.h
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

// The C/F interface header used by the user. This needs a detailed user documentation.

#ifndef HIOP_F_INTERFACE_BASE
#define HIOP_F_INTERFACE_BASE

//include hiop index and size types
#include "hiop_types.h"
#include "FortranCInterface.hpp"
#include <hiop_defs.hpp>


/* Function pointer types for the Fortran callback functions */
typedef void (*f_eval_f_cb)(hiop_size_type* n, double* x, int* new_x, double* obj); 
typedef void (*f_eval_grad_cb)(hiop_size_type* n, double* x, int* new_x, double* gradf); 
typedef void (*f_eval_c_cb)(hiop_size_type* n, hiop_size_type* m, double* x, int* new_x, double* cons); 
typedef void (*f_eval_jac_dense_cb)(hiop_size_type* n,
                                    hiop_size_type* m,
                                    double* x,
                                    int* new_x,
                                    double* mjac);

#ifdef HIOP_SPARSE
typedef void (*f_eval_jac_sparse_cb)(int* task, 
                                     hiop_size_type* n, 
                                     hiop_size_type* m,
                                     double* x,
                                     int* new_x,
                                     hiop_size_type* nnz_jac,
                                     hiop_index_type* irow,
                                     hiop_index_type* jcol,
                                     double* mjac); 
typedef void (*f_eval_hess_sparse_cb)(int* task, 
                                      hiop_size_type* n,
                                      hiop_size_type* m,
                                      double* obj_scal,
                                      double* x,
                                      int* new_x,
                                      double* lambda,
                                      int* new_lam,
                                      hiop_size_type* nnz_hes,
                                      hiop_index_type* irow,
                                      hiop_index_type* jcol,
                                      double* mhes);

typedef struct FSparseProb
{
  hiop_size_type n_;
  hiop_size_type m_;
  hiop_size_type nnz_sparse_Jaceq_;
  hiop_size_type nnz_sparse_Jacineq_;
  hiop_size_type nnz_sparse_Hess_Lagr_;
  double* xlow_;
  double* xupp_;
  double* clow_;
  double* cupp_;
  double* x0_;
  cHiopSparseProblem*   c_prob_; 
  f_eval_f_cb     f_eval_f_;
  f_eval_c_cb     f_eval_c_;
  f_eval_grad_cb  f_eval_grad_;
  f_eval_jac_sparse_cb   f_eval_jac_;
  f_eval_hess_sparse_cb  f_eval_hess_;
} FSparseProb;

int get_prob_sizes_sparse_wrapper(hiop_size_type* n, hiop_size_type* m, void* user_data);
int get_vars_info_sparse_wrapper(hiop_size_type n, double *xlow, double* xupp, void* user_data);
int get_cons_info_sparse_wrapper(hiop_size_type m, double *clow, double* cupp, void* user_data);

void* FC_GLOBAL(hiopsparseprob, HIOPSPARSEPROB) (hiop_size_type*   n,
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
                                                 f_eval_jac_sparse_cb     f_eval_jac,
                                                 f_eval_hess_sparse_cb    f_eval_hess);
void FC_GLOBAL(hiopsparsesolve, HIOPSPARSESOLVE) (void** f_prob_in, double* obj, double* sol);
void FC_GLOBAL(deletehiopsparseprob, DELETEHIOPSPARSEPROB) (void** f_prob_in);

#endif

typedef struct FDenseProb
{
  hiop_size_type n;
  hiop_size_type m;
  double* xlow;
  double* xupp;
  double* clow;
  double* cupp;
  double* x0;
  cHiopDenseProblem*   c_prob; 
  f_eval_f_cb          f_eval_f;
  f_eval_c_cb          f_eval_c;
  f_eval_grad_cb       f_eval_grad;
  f_eval_jac_dense_cb  f_eval_jac;
} FDenseProb;

int get_prob_sizes_dense_wrapper(hiop_size_type* n, hiop_size_type* m, void* user_data);
int get_vars_info_dense_wrapper(hiop_size_type n, double *xlow, double* xupp, void* user_data);
int get_cons_info_dense_wrapper(hiop_size_type m, double *clow, double* cupp, void* user_data);

void* FC_GLOBAL(hiopdenseprob, HIOPDENSEPROB) (hiop_size_type*      n,
                                               hiop_size_type*      m,
                                               double*              xlow,
                                               double*              xupp,
                                               double*              clow,
                                               double*              cupp,
                                               double*              x0,
                                               f_eval_f_cb          f_eval_f,
                                               f_eval_c_cb          f_eval_c,
                                               f_eval_grad_cb       f_eval_grad,
                                               f_eval_jac_dense_cb  f_eval_jac);
void FC_GLOBAL(hiopdensesolve, HIOPDENSESOLVE) (void** f_prob_in, double* obj, double* sol);
void FC_GLOBAL(deletehiopdenseprob, DELETEHIOPDENSEPROB) (void** f_prob_in);

#endif