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
 * @file hiopInterface.h
 *
 * @author Michel Schanen <mschanen@anl.gov>, ANL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

// The C interface header used by the user. This needs a detailed user documentation.

//include hiop index and size types
#include "hiop_types.h"
#include "hiop_defs.hpp"

typedef struct cHiopMDSProblem {
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
} cHiopMDSProblem;
extern int hiop_mds_create_problem(cHiopMDSProblem *problem);
extern int hiop_mds_solve_problem(cHiopMDSProblem *problem);
extern int hiop_mds_destroy_problem(cHiopMDSProblem *problem);

#ifdef HIOP_SPARSE
typedef struct cHiopSparseProblem {
  void *refcppHiop_; // Pointer to the cpp object
  void *hiopinterface_;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void *user_data_; 
  double *solution_;
  double obj_value_;
  int niters_;
  int status_;
  int (*get_starting_point_)(hiop_size_type n, double* x0, void* jprob); 
  int (*get_prob_sizes_)(hiop_size_type* n, hiop_size_type* m, void* jprob); 
  int (*get_vars_info_)(hiop_size_type n, double *xlow, double* xupp, void* jprob);
  int (*get_cons_info_)(hiop_size_type m, double *clow, double* cupp, void* jprob);
  int (*eval_f_)(hiop_size_type n, double* x, int new_x, double* obj, void* jprob);
  int (*eval_grad_f_)(hiop_size_type n, double* x, int new_x, double* gradf, void* jprob);
  int (*eval_cons_)(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* jprob);
  int (*get_sparse_blocks_info_)(hiop_size_type* nx_sparse,
                                 hiop_size_type* nnz_sparse_Jaceq,
                                 hiop_size_type* nnz_sparse_Jacineq,
                                 hiop_size_type* nnz_sparse_Hess_Lagr_SS,
                                 void* jprob);
  int (*eval_Jac_cons_)(hiop_size_type n, 
                        hiop_size_type m,
                        double* x,
                        int new_x,
                        hiop_size_type nnzJacS,
                         hiop_index_type* iJacS,
                        hiop_index_type* jJacS,
                        double* MJacS,
                        void *jprob);
  int (*eval_Hess_Lagr_)(hiop_size_type n,
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
                         void* jprob);
} cHiopSparseProblem;

extern int hiop_sparse_create_problem(cHiopSparseProblem *problem);
extern int hiop_sparse_solve_problem(cHiopSparseProblem *problem);
extern int hiop_sparse_destroy_problem(cHiopSparseProblem *problem);
#endif

typedef struct cHiopDenseProblem {
  void *refcppHiop; // Pointer to the cpp object
  void *hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void *user_data; 
  double *solution;
  double obj_value;
  int niters;
  int status;
  int (*get_starting_point)(hiop_size_type n_, double* x0, void* jprob); 
  int (*get_prob_sizes)(hiop_size_type* n_, hiop_size_type* m_, void* jprob); 
  int (*get_vars_info)(hiop_size_type n, double *xlow_, double* xupp_, void* jprob);
  int (*get_cons_info)(hiop_size_type m, double *clow_, double* cupp_, void* jprob);
  int (*eval_f)(hiop_size_type n, double* x, int new_x, double* obj, void* jprob);
  int (*eval_grad_f)(hiop_size_type n, double* x, int new_x, double* gradf, void* jprob);
  int (*eval_cons)(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* jprob);
  int (*eval_Jac_cons)(hiop_size_type n, 
                       hiop_size_type m,
                       double* x,
                       int new_x,
                       double* MJac,
                       void *jprob);
} cHiopDenseProblem;

extern int hiop_dense_create_problem(cHiopDenseProblem *problem);
extern int hiop_dense_solve_problem(cHiopDenseProblem *problem);
extern int hiop_dense_destroy_problem(cHiopDenseProblem *problem);
