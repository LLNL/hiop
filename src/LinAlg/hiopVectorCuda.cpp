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
 * @file hiopVectorCuda.cpp
 *
 * @author Nai-Yuan Chiabg <chiang7@llnl.gov>, LLNL
 *
 */
#include "hiopVectorCuda.hpp"
#include "hiopVectorIntCuda.hpp"
#include "VectorCudaKernels.hpp"
#include "MathDeviceKernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <limits>

namespace hiop
{

hiopVectorCuda::hiopVectorCuda(const size_type& glob_n, index_type* col_part, MPI_Comm comm)
  : hiopVector(),
    comm_(comm)
{
  n_ = glob_n;

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_ == MPI_COMM_NULL) 
    comm_ = MPI_COMM_SELF;
#endif

  int P = 0; 
  if(col_part)
  {
#ifdef HIOP_USE_MPI
    int ierr=MPI_Comm_rank(comm_, &P);  assert(ierr==MPI_SUCCESS);
#endif
    glob_il_ = col_part[P];
    glob_iu_ = col_part[P+1];
  } 
  else
  {
    glob_il_ = 0;
    glob_iu_ = n_;
  }
  n_local_ = glob_iu_ - glob_il_;

  // Size in bytes
  size_t bytes = n_local_ * sizeof(double);
  
  // Allocate memory on host
  data_host_ = new double[n_local_];
 
  // Allocate memory on GPU
  cudaError_t cuerr = cudaMalloc((void**)&data_dev_, bytes);
  assert(cudaSuccess == cuerr);

  // handles
  cublasCreate(&handle_cublas_);
}

hiopVectorCuda::hiopVectorCuda(const hiopVectorCuda& v)
 : hiopVector()
{
  n_local_ = v.n_local_;
  n_ = v.n_;
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;

  // Size in bytes
  size_t bytes = n_local_ * sizeof(double);
  
  // Allocate memory on host
  data_host_ = new double[bytes];
 
  // Allocate memory on GPU
  cudaError_t cuerr = cudaMalloc((void**)&data_dev_, bytes);
  assert(cudaSuccess == cuerr);

  // handles
  cublasCreate(&handle_cublas_);
}

hiopVectorCuda::~hiopVectorCuda()
{
  delete [] data_host_;

  // Delete workspaces and handles
  cudaFree(data_dev_);
  cublasDestroy(handle_cublas_);
}

/// Set all vector elements to zero
void hiopVectorCuda::setToZero()
{
  hiop::cuda::thrust_fill_kernel(n_local_, data_dev_, 0.0);
}

/// Set all vector elements to constant c
void hiopVectorCuda::setToConstant(double c)
{
  hiop::cuda::thrust_fill_kernel(n_local_, data_dev_, c);
}

/// Set all elements to random values uniformly distributed between `minv` and `maxv`.
void hiopVectorCuda::set_to_random_uniform(double minv, double maxv)
{
  double* data = data_dev_;
  hiop::device::array_random_uniform_kernel(n_local_, data, minv, maxv);
} // namespace hiop

/// Set all elements that are not zero in ix to  c, and the rest to 0
void hiopVectorCuda::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorCuda& ix = dynamic_cast<const hiopVectorCuda&>(select);
  // TODO: add one cu function to perform the following two functions
  setToConstant(c);
  componentMult(select);
}

void hiopVectorCuda::copyFrom(const hiopVector& v_)
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(v_);
  cudaError_t cuerr = cudaMemcpy(data_dev_, v.data_dev_, (n_local_)*sizeof(double), cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyFrom(const double* v_local_data)
{
  if(v_local_data) {
    cudaError_t cuerr = cudaMemcpy(data_dev_, v_local_data, (n_local_)*sizeof(double), cudaMemcpyDeviceToDevice);
    assert(cuerr == cudaSuccess);
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorCuda::copy_from_w_pattern(const hiopVector& vv, const hiopVector& select)
{
  const hiopVectorCuda& ix = dynamic_cast<const hiopVectorCuda&>(select); 
  copyFrom(vv);
  componentMult(select);
}

void hiopVectorCuda::copyFromStarting(int start_index_in_dest, const double* v, int nv)
{
  assert(start_index_in_dest+nv <= n_local_);
  cudaError_t cuerr = cudaMemcpy(data_dev_+start_index_in_dest, v, (nv)*sizeof(double), cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyFromStarting(int start_index_in_dest, const hiopVector& v_src)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_==n_ && "only for local/non-distributed vectors");
#endif
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(v_src);
  assert(start_index_in_dest+v.n_local_ <= n_local_);
  cudaError_t cuerr = cudaMemcpy(data_dev_+start_index_in_dest, v.data_dev_, (v.n_local_)*sizeof(double), cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copy_from_starting_at(const double* v, int start_index_in_v, int nv)
{
  cudaError_t cuerr = cudaMemcpy(data_dev_, v+start_index_in_v, (nv)*sizeof(double), cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src) 
{
  const hiopVectorIntCuda& idx_src = dynamic_cast<const hiopVectorIntCuda&>(index_in_src);
  const hiopVectorCuda& v_src = dynamic_cast<const hiopVectorCuda&>(src); 
  assert(idx_src.size() == n_local_);

  int* id = const_cast<int*>(idx_src.local_data_const());
  double* dd = data_dev_;
  const double* vd = v_src.local_data_const();
  
  hiop::cuda::copy_from_index_kernel(n_local_, dd, vd, id);
}

void hiopVectorCuda::copy_from_indexes(const double* src, const hiopVectorInt& index_in_src)
{
  const hiopVectorIntCuda& idx_src = dynamic_cast<const hiopVectorIntCuda&>(index_in_src);
  assert(idx_src.size() == n_local_);

  int* id = const_cast<int*>(idx_src.local_data_const());
  double* dd = data_dev_;
  
  hiop::cuda::copy_from_index_kernel(n_local_, dd, src, id);
}

void hiopVectorCuda::startingAtCopyFromStartingAt(int start_idx_dest,
                                                  const hiopVector& vec_src,
                                                  int start_idx_src)
{
  size_type howManyToCopyDest = this->n_local_ - start_idx_dest;

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif

  assert((start_idx_dest >= 0 && start_idx_dest < this->n_local_) || this->n_local_==0);
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec_src);
  assert((start_idx_src >=0 && start_idx_src < v.n_local_) || v.n_local_==0 || v.n_local_==start_idx_src);
  const size_type howManyToCopySrc = v.n_local_-start_idx_src;  

  if(howManyToCopyDest == 0 || howManyToCopySrc == 0) {
    return;
  }

  assert(howManyToCopyDest <= howManyToCopySrc);

  cudaError_t cuerr = cudaMemcpy(data_dev_ + start_idx_dest,
                                 v.data_dev_ + start_idx_src,
                                 (howManyToCopyDest)*sizeof(double),
                                 cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

/// @brief Copy `this` vector local data to `dest` buffer.
void hiopVectorCuda::copyTo(double* dest) const
{
  cudaError_t cuerr = cudaMemcpy(dest,
                                 data_dev_,
                                 (n_local_)*sizeof(double),
                                 cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyToStarting(int start_index, hiopVector& dst) const
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(dst);

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert(start_index + v.n_local_ <= n_local_);

  // If nowhere to copy, return.
  if(v.n_local_ == 0)
    return;

  cudaError_t cuerr = cudaMemcpy(v.data_dev_,
                                 data_dev_ + start_index,
                                 (v.n_local_)*sizeof(double),
                                 cudaMemcpyDeviceToDevice);
  assert(cuerr = cudaSuccess);
}

void hiopVectorCuda::copyToStarting(hiopVector& vec, int start_index_in_dest) const
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec);
  assert(start_index_in_dest+n_local_ <= v.n_local_);

  // If there is nothing to copy, return.
  if(n_local_ == 0)
    return;

  cudaError_t cuerr = cudaMemcpy(v.data_dev_ + start_index_in_dest,
                                 data_dev_,
                                 (n_local_)*sizeof(double),
                                 cudaMemcpyDeviceToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyToStartingAt_w_pattern(hiopVector& v, int start_index_in_dest, const hiopVector& ix) const
{
  assert(false && "not needed / implemented");
}

void hiopVectorCuda::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                 const hiopVectorInt& c_map,
                                                 const hiopVector& d,
                                                 const hiopVectorInt& d_map)
{
  const int c_size = c.get_size();
  const int d_size = d.get_size();

  assert( c_size == c_map.size() );
  assert( d_size == d_map.size() );
  assert( c_size + d_size == n_local_);

  hiop::device::copy_src_to_mapped_dest_kernel(c_size, c.local_data_const(), local_data(), c_map.local_data_const());
  hiop::device::copy_src_to_mapped_dest_kernel(d_size, d.local_data_const(), local_data(), d_map.local_data_const());
}

void hiopVectorCuda::copy_to_two_vec_w_pattern(hiopVector& c,
                                               const hiopVectorInt& c_map,
                                               hiopVector& d,
                                               const hiopVectorInt& d_map) const
{
  const int c_size = c.get_size();
  const int d_size = d.get_size();

  assert( c_size == c_map.size() );
  assert( d_size == d_map.size() );
  assert( c_size + d_size == n_local_);

  hiop::device::copy_mapped_src_to_dest_kernel(c_size, local_data_const(), c.local_data(), c_map.local_data_const());
  hiop::device::copy_mapped_src_to_dest_kernel(d_size, local_data_const(), d.local_data(), d_map.local_data_const());
}

void hiopVectorCuda::startingAtCopyToStartingAt(int start_idx_in_src, 
                                                hiopVector& destination, 
                                                int start_idx_dest, 
                                                int num_elems /* = -1 */) const
{
  assert(false&&"TODO");
}

void hiopVectorCuda::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                          hiopVector& destination,
                                                          index_type start_idx_dest,
                                                          const hiopVector& selec_dest,
                                                          size_type num_elems/*=-1*/) const
{
  assert(false&&"TODO");
}
 
double hiopVectorCuda::twonorm() const
{
  int one = 1; 
  double nrm = 0.;
  if(n_local_>0) {
    cublasStatus_t ret_cublas = cublasDnrm2(handle_cublas_, n_local_, data_dev_, one, &nrm);
    assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
  }

#ifdef HIOP_USE_MPI
  nrm *= nrm;
  double nrmG;
  int ierr = MPI_Allreduce(&nrm, &nrmG, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  nrm = std::sqrt(nrmG);
#endif  
  return nrm;
}

double hiopVectorCuda::infnorm() const
{
  double nrm = infnorm_local();
#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(MPI_SUCCESS==ierr);
  return nrm_global;
#endif

  return nrm;
}

double hiopVectorCuda::infnorm_local() const
{
  return hiop::cuda::infnorm_local_kernel(n_local_, data_dev_);
}

double hiopVectorCuda::onenorm() const
{
  double norm1 = onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&norm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  return nrm1_global;
#endif
  return norm1;
}

double hiopVectorCuda::onenorm_local() const
{
//  double* data = data_dev_;
//  int n = n_local_;
//  return hiop::cuda::onenorm_local_kernel(n, data);
    return hiop::cuda::onenorm_local_kernel(n_local_, data_dev_);
}

void hiopVectorCuda::componentMult( const hiopVector& vec )
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec);
  assert(n_local_ == v.n_local_);

  hiop::cuda::thrust_component_mult_kernel(n_local_, data_dev_, v.data_dev_);
}

void hiopVectorCuda::componentDiv( const hiopVector& vec )
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec);
  assert(n_local_ == v.n_local_);

  hiop::cuda::thrust_component_div_kernel(n_local_, data_dev_, v.data_dev_);
}

void hiopVectorCuda::componentDiv_w_selectPattern( const hiopVector& vec, const hiopVector& select)
{
  const hiopVectorCuda& ix = dynamic_cast<const hiopVectorCuda&>(select);
  hiop::cuda::component_div_w_pattern_kernel(n_local_, data_dev_, v.data_dev_, ix.data);
}

void hiopVectorCuda::component_min(const double constant)
{
  hiop::cuda::component_min_kernel(n_local_, data_dev_, constant);
}

void hiopVectorCuda::component_min(const hiopVector& vec)
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec); 
  assert(v.get_local_size() == n_local_);

  const double* vd = v.data_dev_;
  
  hiop::cuda::component_min_kernel(n_local_, data_dev_, vd);
}

void hiopVectorCuda::component_max(const double constant)
{
  hiop::cuda::component_max_kernel(n_local_, data_dev_, constant);
}

void hiopVectorCuda::component_max(const hiopVector& vec)
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec); 
  assert(v.get_local_size() == n_local_);

  const double* vd = v.data_dev_;
  
  hiop::cuda::component_max_kernel(n_local_, data_dev_, vd);
}

void hiopVectorCuda::component_abs()
{
  hiop::cuda::thrust_component_abs_kernel(n_local_, data_dev_);
}

void hiopVectorCuda::component_sgn ()
{
  hiop::cuda::thrust_component_sgn_kernel(n_local_, data_dev_);
}

void hiopVectorCuda::component_sqrt()
{
  hiop::cuda::thrust_component_sqrt_kernel(n_local_, data_dev_);
}

/// Scale each element of this  by the constant alpha
void hiopVectorCuda::scale(double alpha)
{
  int one = 1;  
  cublasStatus_t ret_cublas = cublasDscal(handle_cublas_, n_local_, &alpha, data_dev_, one);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
}

/// Implementation of AXPY kernel
void hiopVectorCuda::axpy(double alpha, const hiopVector& xvec)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  int one = 1;
  cublasStatus_t ret_cublas = cublasDaxpy(handle_cublas_, n_local_, &alpha, x.data_dev_, one, data_dev_, one);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
}

/// this += alpha * x, for the entries in 'this' where corresponding 'select' is nonzero.
void hiopVectorCuda::axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select) 
{
  const hiopVectorCuda& ix = dynamic_cast<const hiopVectorCuda&>(select);
  axpy(alpha, xvec);
  componentMult(select);
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void hiopVectorCuda::axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  const hiopVectorIntCuda& idxs = dynamic_cast<const hiopVectorIntCuda&>(i);

  assert(x.get_size()==i.size());
  assert(x.get_local_size()==i.size());
  assert(i.size()<=n_local_);
  
  double* yd = data_dev_;
  double* xd = const_cast<double*>(x.data_dev_);
  int* id = const_cast<int*>(idxs.local_data_const());

  hiop::cuda::axpy_w_map_kernel(n_local_, yd, xd, id, alpha);
}

/** @brief this[i] += alpha*x[i]*z[i] forall i */
void hiopVectorCuda::axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  const hiopVectorCuda& z = dynamic_cast<const hiopVectorCuda&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(  n_local_ == z.n_local_);
#endif  
  double* dd       = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();

  hiop::cuda::axzpy_kernel(n_local_, dd, xd, zd, alpha);
}

/** @brief this[i] += alpha*x[i]/z[i] forall i */
void hiopVectorCuda::axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  const hiopVectorCuda& z = dynamic_cast<const hiopVectorCuda&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(  n_local_ == z.n_local_);
#endif  
  double*       yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();

  hiop::cuda::axdzpy_kernel(n_local_, yd, xd, zd, alpha);
}

/** @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection */
void hiopVectorCuda::axdzpy_w_pattern(double alpha,
                                      const hiopVector& xvec,
                                      const hiopVector& zvec,
                                      const hiopVector& select)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  const hiopVectorCuda& z = dynamic_cast<const hiopVectorCuda&>(zvec);
  const hiopVectorCuda& sel = dynamic_cast<const hiopVectorCuda&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==z.n_local_);
  assert(  n_local_==z.n_local_);
#endif
  double* yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();
  const double* id = sel.local_data_const();

  hiop::cuda::axdzpy_w_pattern_kernel(n_local_, yd, xd, zd, id, alpha);
}

/** @brief this[i] += c forall i */
void hiopVectorCuda::addConstant(double c)
{
  hiop::cuda::add_constant_kernel(n_local_, data_dev_, c);
}

/** @brief this[i] += c forall i with pattern selection */
void  hiopVectorCuda::addConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorCuda& sel = dynamic_cast<const hiopVectorCuda&>(select);
  assert(this->n_local_ == sel.n_local_);
  const double* id = sel.local_data_const();

  hiop::cuda::add_constant_w_pattern_kernel(n_local_, data_dev_, id, c);
}

/** Return the dot product of this hiopVector with v */
double hiopVectorCuda::dotProductWith( const hiopVector& v ) const
{
  const hiopVectorCuda& vx = dynamic_cast<const hiopVectorCuda&>(v);
  int one = 1;
  double retval; 
  cublasStatus_t ret_cublas = cublasDdot(handle_cublas_, n_local_, vx.data_dev_, one, data_dev_, one, &retval);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
  return retval;
}

/// @brief Negate all the elements of this
void hiopVectorCuda::negate()
{
  hiop::cuda::thrust_negate_kernel(n_local_, data_dev_);
}

/// @brief Invert (1/x) the elements of this
void hiopVectorCuda::invert()
{
  hiop::cuda::invert_kernel(n_local_, data_dev_);
}

/** @brief Sum all selected log(this[i]) */
double hiopVectorCuda::logBarrier_local(const hiopVector& select) const
{
  const hiopVectorCuda& sel = dynamic_cast<const hiopVectorCuda&>(select);
  assert(n_local_ == sel.n_local_);
  const double* id = sel.local_data_const();

  return hiop::cuda::log_barr_obj_kernel(n_local_, data_dev_, id);
}

/**  @brief add 1/(this[i]) */
void hiopVectorCuda::addLogBarrierGrad(double alpha,
                                       const hiopVector& xvec,
                                       const hiopVector& select)
{
  const hiopVectorCuda& x = dynamic_cast<const hiopVectorCuda&>(xvec);
  const hiopVectorCuda& sel = dynamic_cast<const hiopVectorCuda&>(select);  
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == x.n_local_);
  assert(n_local_ == sel.n_local_);
#endif
  double* yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* id = sel.local_data_const();

  hiop::cuda::adxpy_w_pattern_kernel(n_local_, yd, xd, id, alpha);
}

/** @brief Sum all elements */
double hiopVectorCuda::sum_local() const
{
  return hiop::cuda::thrust_sum_kernel(n_local_, data_dev_);
}

/**
 * @brief Linear damping term */
double hiopVectorCuda::linearDampingTerm_local(const hiopVector& ixleft,
                                               const hiopVector& ixright,
                                               const double& mu,
                                               const double& kappa_d) const
{
  const hiopVectorCuda& ixl = dynamic_cast<const hiopVectorCuda&>(ixleft);
  const hiopVectorCuda& ixr = dynamic_cast<const hiopVectorCuda&>(ixright);  
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == ixl.n_local_);
  assert(n_local_ == ixr.n_local_);
#endif
  const double* ld = ixl.local_data_const();
  const double* rd = ixr.local_data_const();
  const double* vd = data_dev_;

  return hiop::cuda::linear_damping_term_kernel(n_local_, vd, ld, rd, mu, kappa_d);
}

void hiopVectorCuda::addLinearDampingTerm(const hiopVector& ixleft,
                                          const hiopVector& ixright,
                                          const double& alpha,
                                          const double& ct)
{

  assert((dynamic_cast<const hiopVectorCuda&>(ixleft)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorCuda&>(ixright)).n_local_ == n_local_);

  const double* ixl= (dynamic_cast<const hiopVectorCuda&>(ixleft)).local_data_const();
  const double* ixr= (dynamic_cast<const hiopVectorCuda&>(ixright)).local_data_const();

  double* data = data_dev_;

  // compute linear damping term
  hiop::cuda::add_linear_damping_term_kernel(n_local_, data, ixl, ixr, alpha, ct);
}

/** @brief Check if all elements of the vector are positive */
int hiopVectorCuda::allPositive()
{
  double min_val = hiop::cuda::min_local_kernel(n_local_, data_dev_);

  int allPos = (min_val > 0.0) ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

/** @brief Checks if selected elements of `this` are positive */
int hiopVectorCuda::allPositive_w_patternSelect(const hiopVector& wvec)
{
  const hiopVectorCuda& w = dynamic_cast<const hiopVectorCuda&>(wvec);

#ifdef HIOP_DEEPCHECKS
  assert(w.n_local_ == n_local_);
#endif 

  const double* id = w.local_data_const();
  const double* data = data_dev_;

  int allPos = hiop::cuda::all_positive_w_pattern_kernel(n_local_, data, id);
  
  allPos = (allPos==n_local_) ? 1 : 0;
  
#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif  
  return allPos;
}

/// Find minimum vector element
double hiopVectorCuda::min() const
{
  double result = hiop::cuda::min_local_kernel(n_local_, data_dev_);

#ifdef HIOP_USE_MPI
  double resultG;
  double ierr=MPI_Allreduce(&result, &resultG, 1, MPI_DOUBLE, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return resultG;
#endif
  return result;
}

/// Find minimum vector element for `select` pattern
double hiopVectorCuda::min_w_pattern(const hiopVector& select) const
{
  const hiopVectorCuda& sel = dynamic_cast<const hiopVectorCuda&>(select);
  assert(this->n_local_ == sel.n_local_);
  const double* data = data_dev_;
  const double* id = sel.local_data_const();
  
  double max_val = std::numeric_limits<double>::max();
  double result = hiop::cuda::min_w_pattern_kernel(n_local_, data, id, max_val);

#ifdef HIOP_USE_MPI
  double resultG;
  double ierr=MPI_Allreduce(&result, &resultG, 1, MPI_DOUBLE, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return resultG;
#endif
  return result;
}

/// Find minimum vector element. TODO
void hiopVectorCuda::min( double& /* m */, int& /* index */) const
{
  assert(false && "not implemented");
}

/** @brief Project solution into bounds  */
bool hiopVectorCuda::projectIntoBounds_local(const hiopVector& xlo, 
                                             const hiopVector& ixl,
                                             const hiopVector& xup,
                                             const hiopVector& ixu,
                                             double kappa1,
                                             double kappa2)
{
  const hiopVectorCuda& xl = dynamic_cast<const hiopVectorCuda&>(xlo);
  const hiopVectorCuda& il = dynamic_cast<const hiopVectorCuda&>(ixl);
  const hiopVectorCuda& xu = dynamic_cast<const hiopVectorCuda&>(xup);
  const hiopVectorCuda& iu = dynamic_cast<const hiopVectorCuda&>(ixu);

#ifdef HIOP_DEEPCHECKS
  assert(xl.n_local_ == n_local_);
  assert(il.n_local_ == n_local_);
  assert(xu.n_local_ == n_local_);
  assert(iu.n_local_ == n_local_);
#endif

  const double* xld = xl.local_data_const();
  const double* ild = il.local_data_const();
  const double* xud = xu.local_data_const();
  const double* iud = iu.local_data_const();
  double* xd = data_dev_;
  
  // Perform preliminary check to see of all upper value
  bool bval = hiop::cuda::check_bounds_kernel(n_local_, xld, xud);

  if(false == bval) 
    return false;

  const double small_real = std::numeric_limits<double>::min() * 100;
  
  hiop::cuda::project_into_bounds_kernel(n_local_, xd, xld, ild, xud, iud, kappa1, kappa2, small_real);

  return true;
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorCuda::fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const
{
  const hiopVectorCuda& d = dynamic_cast<const hiopVectorCuda&>(dvec);
#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1); // TODO: per documentation above it should be tau <= 1 (?).
#endif

  const double* dd = d.local_data_const();
  const double* xd = data_dev_;

  double alpha = hiop::cuda::min_frac_to_bds_kernel(n_local_, xd, dd, tau);

  return alpha;
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select */
double hiopVectorCuda::fractionToTheBdry_w_pattern_local(const hiopVector& dvec,
                                                         const double& tau, 
                                                         const hiopVector& select) const
{
  const hiopVectorCuda& d = dynamic_cast<const hiopVectorCuda&>(dvec);
  const hiopVectorCuda& s = dynamic_cast<const hiopVectorCuda&>(select);

#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(s.n_local_ == n_local_);
  assert(tau>0);
  assert(tau<1);
#endif
  const double* dd = d.local_data_const();
  const double* xd = data_dev_;
  const double* id = s.local_data_const();

  double alpha = hiop::cuda::min_frac_to_bds_w_pattern_kernel(n_local_, xd, dd, id, tau);

  return alpha;
}

/** @brief Set elements of `this` to zero based on `select`.*/
void hiopVectorCuda::selectPattern(const hiopVector& select)
{
  const hiopVectorCuda& s = dynamic_cast<const hiopVectorCuda&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(s.n_local_==n_local_);
#endif

  double* data = data_dev_;
  double* id = s.data_dev_;

  // set value with pattern
  hiop::cuda::select_pattern_kernel(n_local_, data, id);
}

/** @brief Checks if `this` matches nonzero pattern of `select`. */
bool hiopVectorCuda::matchesPattern(const hiopVector& pattern)
{  
  const hiopVectorCuda& p = dynamic_cast<const hiopVectorCuda&>(pattern);

#ifdef HIOP_DEEPCHECKS
  assert(p.n_local_==n_local_);
#endif

  double* xd = data_dev_;
  double* id = p.data_dev_;

  return hiop::cuda::match_pattern_kernel(n_local_, xd, id);
}

/** @brief Adjusts duals. */
void hiopVectorCuda::adjustDuals_plh(const hiopVector& xvec, 
                                     const hiopVector& ixvec,
                                     const double& mu,
                                     const double& kappa)
{
  const hiopVectorCuda& x  = dynamic_cast<const hiopVectorCuda&>(xvec) ;
  const hiopVectorCuda& ix = dynamic_cast<const hiopVectorCuda&>(ixvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==n_local_);
  assert(ix.n_local_==n_local_);
#endif
  const double* xd =  x.local_data_const();
  const double* id = ix.local_data_const();
  double* zd = data_dev_; //the dual

  hiop::cuda::adjustDuals_plh_kernel(n_local_, zd, xd, id, mu, kappa);
}

/** @brief Check if all elements of the vector are zero */
bool hiopVectorCuda::is_zero() const
{
  return hiop::cuda::is_zero_kernel(n_local_, data_dev_);
}

/** @brief Returns true if any element of `this` is NaN. */
bool hiopVectorCuda::isnan_local() const
{
  return hiop::cuda::isnan_kernel(n_local_, data_dev_);
}

/**
 * @brief Returns true if any element of `this` is Inf.
 * 
 * @post `this` is not modified
 * 
 * @warning This is local method only!
 */
bool hiopVectorCuda::isinf_local() const
{
  return hiop::cuda::isinf_kernel(n_local_, data_dev_);
}

/** @brief Returns true if all elements of `this` are finite. */
bool hiopVectorCuda::isfinite_local() const
{
  return hiop::cuda::isfinite_kernel(n_local_, data_dev_);
}

/** @brief Prints vector data to a file in Matlab format. */
void hiopVectorCuda::print(FILE* file/*=nullptr*/, const char* msg/*=nullptr*/, int max_elems/*=-1*/, int rank/*=-1*/) const
{
  // TODO. no fprintf. use printf to print everything on screen?
  assert(false && "Not implemented in CUDA device");
}

hiopVector* hiopVectorCuda::alloc_clone() const
{
  hiopVector* v = new hiopVectorCuda(*this); assert(v);
  return v;
}

hiopVector* hiopVectorCuda::new_copy () const
{
  hiopVector* v = new hiopVectorCuda(*this); assert(v);
  v->copyFrom(*this);
  return v;
}

void hiopVectorCuda::copyToDev()
{
  cudaError_t cuerr = cudaMemcpy(data_dev_, data_host_, (n_local_)*sizeof(double), cudaMemcpyHostToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyFromDev()
{
  cudaError_t cuerr = cudaMemcpy(data_host_, data_dev_, (n_local_)*sizeof(double), cudaMemcpyDeviceToHost);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyToDev() const
{
  double* data_dev = const_cast<double*>(data_dev_);
  cudaError_t cuerr = cudaMemcpy(data_dev, data_host_, (n_local_)*sizeof(double), cudaMemcpyHostToDevice);
  assert(cuerr == cudaSuccess);
}

void hiopVectorCuda::copyFromDev() const
{
  double* data_host = const_cast<double*>(data_host_);
  cudaError_t cuerr = cudaMemcpy(data_host, data_dev_, (n_local_)*sizeof(double), cudaMemcpyDeviceToHost);
  assert(cuerr == cudaSuccess);
}

size_type hiopVectorCuda::numOfElemsLessThan(const double &val) const
{
  return hiop::cuda::num_of_elem_less_than_kernel(n_local_, data_dev_, val);
}

size_type hiopVectorCuda::numOfElemsAbsLessThan(const double &val) const
{
  return hiop::cuda::num_of_elem_absless_than_kernel(n_local_, data_dev_, val);
}

void hiopVectorCuda::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                          const int start, 
                                          const int end, 
                                          const hiopInterfaceBase::NonlinearityType* arr_src,
                                          const int start_src) const
{
  assert(arr && arr_src);
  assert(end <= n_local_ && start <= end && start >= 0);
  // If there is nothing to copy, return.
  const int length = end - start;
  if(length == 0) {
    return;
  }

  hiop::cuda::set_array_from_to_kernel(n_local_, arr, start, length, arr_src, start_src);
}

void hiopVectorCuda::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                       const int start, 
                                       const int end, 
                                       const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(arr && arr_src);
  assert(end <= n_local_ && start <= end && start >= 0);
  // If there is nothing to copy, return.
  int length = end - start;
  if(length == 0) {
    return;
  }

  hiop::cuda::set_array_from_to_kernel(n_local_, arr, start, length, arr_src);
}






bool hiopVectorCuda::is_equal(const hiopVector& vec) const
{
  assert(false&&"NOT needed. Remove this func. TODO");
}



} // namespace hiop

