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
 * @file VectorCudaKernels.hpp
 *
 * @author Nai-Yuan Chiabg <chiang7@llnl.gov>, LLNL
 *
 */
#ifndef HIOP_VECTOR_CUDA_KER
#define HIOP_VECTOR_CUDA_KER

#include <thrust/functional.h>
#include "hiopInterface.hpp"


namespace hiop
{
namespace cuda
{

void copy_from_index_kernel(int n_local,
                            double* yd,
                            const double* src,
                            const int* id);

void component_min_kernel(int n_local,
                          double* yd,
                          double c);

void component_min_kernel(int n_local,
                          double* yd,
                          const double* xd);

void component_max_kernel(int n_local,
                          double* yd,
                          double c);

void component_max_kernel(int n_local,
                          double* yd,
                          const double* xd);

void axpy_w_map_kernel(int n_local,
                       double* yd,
                       const double* xd,
                       const int* id,
                       double alpha);

void axzpy_kernel(int n_local,
                  double* yd,
                  const double* xd,
                  const double* zd,
                  double alpha);

void axdzpy_kernel(int n_local,
                   double* yd,
                   const double* xd,
                   const double* zd,
                   double alpha);

void axdzpy_w_pattern_kernel(int n_local,
                             double* yd,
                             const double* xd,
                             const double* zd,
                             const double* id,
                             double alpha);

void add_constant_kernel(int n_local, double* yd, double c);

void  add_constant_w_pattern_kernel(int n_local, double* yd, const double* id, double c);

void invert_kernel(int n_local, double* yd);

void adxpy_w_pattern_kernel(int n_local,
                            double* yd,
                            const double* xd,
                            const double* ld,
                             double alpha);

void component_div_w_pattern_kernel(int n_local,
                                    double* yd,
                                    const double* xd,
                                    const double* id);

void set_linear_damping_term_kernel(int n_local,
                                    double* yd,
                                    const double* vd,
                                    const double* ld,
                                    const double* rd);

void add_linear_damping_term_kernel(int n_local,
                                    double* yd,
                                    const double* ixl,
                                    const double* ixr,
                                    double alpha,
                                    double ct);

void is_posive_w_pattern_kernel(int n_local,
                                double* yd,
                                const double* xd,
                                const double* id);

void set_val_w_pattern_kernel(int n_local,
                              double* yd,
                              const double* xd,
                              const double* id,
                              double max_val);

void project_into_bounds_kernel(int n_local,
                                double* xd,
                                const double* xld,
                                const double* ild,
                                const double* xud,
                                const double* iud,
                                double kappa1,
                                double kappa2,
                                double small_real);

void fraction_to_the_boundry_kernel(int n_local,
                                    double* yd,
                                    const double* xd,
                                    const double* dd,
                                    double tau);

void fraction_to_the_boundry_w_pattern_kernel(int n_local,
                                              double* yd,
                                              const double* xd,
                                              const double* dd,
                                              const double* id,
                                              double tau);

void select_pattern_kernel(int n_local, double* yd, const double* id);

void component_match_pattern_kernel(int n_local, int* yd, const double* xd, const double* id);

void adjustDuals_plh_kernel(int n_local,
                            double* yd,
                            const double* xd,
                            const double* id,
                            double mu,
                            double kappa);

void set_array_from_to_kernel(int n_local, 
                              hiop::hiopInterfaceBase::NonlinearityType* arr, 
                              int start, 
                              int length, 
                              const hiop::hiopInterfaceBase::NonlinearityType* arr_src,
                              int start_src);

void set_array_from_to_kernel(int n_local, 
                              hiop::hiopInterfaceBase::NonlinearityType* arr, 
                              int start, 
                              int length,
                              hiop::hiopInterfaceBase::NonlinearityType arr_src);


void thrust_fill_kernel(int n, double* ptr, double c);

double infnorm_local_kernel(int n, double* data_dev);
double onenorm_local_kernel(int n, double* data_dev);

void thrust_component_mult_kernel(int n, double* d1, double* d2);
void thrust_component_div_kernel(int n, double* d1, double* d2);
void thrust_component_abs_kernel(int n, double* d1);
void thrust_component_sgn_kernel(int n, double* d1);
void thrust_component_sqrt_kernel(int n, double* d1);
void thrust_negate_kernel(int n, double* d1);
double log_barr_obj_kernel(int n, double* d1, const double* id);
double thrust_sum_kernel(int n, double* d1);
double linear_damping_term_kernel(int n,
                                  const double* vd,
                                  const double* ld,
                                  const double* rd,
                                  double mu,
                                  double kappa_d);
double min_local_kernel(int n, double* d1);
int all_positive_w_pattern_kernel(int n, const double* d1, const double* id);
double min_w_pattern_kernel(int n, const double* d1, const double* id, double max_val);
bool check_bounds_kernel(int n, const double* xld, const double* xud);
double min_frac_to_bds_kernel(int n, const double* xd, const double* dd, double tau);
double min_frac_to_bds_w_pattern_kernel(int n,
                                        const double* xd,
                                        const double* dd,
                                        const double* id,
                                        double tau);
bool match_pattern_kernel(int n, const double* xd, const double* id);
bool is_zero_kernel(int n, double* xd);
bool isnan_kernel(int n, double* xd);
bool isinf_kernel(int n, double* xd);
bool isfinite_kernel(int n, double* xd);
int num_of_elem_less_than_kernel(int n, double* xd, double val);
int num_of_elem_absless_than_kernel(int n, double* xd, double val);


/// for hiopVectorIntCuda
void set_to_linspace_kernel(int sz, int* buf, int i0, int di);
void compute_cusum_kernel(int sz, int* buf, const double* id);
void copyToStartingAt_w_pattern_kernel(int n_src, 
                                       int n_dest,
                                       int start_index_in_dest,
                                       int* nnz_cumsum, 
                                       double *vd,
                                       const double* dd);
}
}
#endif

