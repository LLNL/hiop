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
#include "MathDeviceKernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/replace.h>
#include <thrust/transform_reduce.h>

#include <thrust/execution_policy.h>

/// @brief  operators for thurst
template <typename T>
struct thrust_abs: public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& a)
    {
        return fabs(a);
    }
};

template <typename T>
struct thrust_abs_diff: public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return fabs(b - a);
    }
};

namespace hiop
{

hiopVectorCuda::hiopVectorCuda(const size_type& glob_n, std::string mem_space, index_type* col_part, MPI_Comm comm)
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
  data_host_ = new double[bytes];
 
  // Allocate memory on GPU
  cudaError_t cuerr = cudaMalloc(&data_dev_, bytes);
  assert(cudaSuccess == cuerr);

  // handles
  cublasCreate(&handle_cublas_);
}

hiopVectorCuda::~hiopVectorCuda()
{
  delete data_host_;

  // Delete workspaces and handles
  cudaFree(data_dev_);
  cublasDestroy(handle_cublas_);
}

/// Set all vector elements to zero
void hiopVectorCuda::setToZero()
{
  hiop::device::set_to_val_kernel(n_local_, data_dev_, 0.0);
}

/// Set all vector elements to constant c
void hiopVectorCuda::setToConstant(double c)
{
  hiop::device::set_to_val_kernel(n_local_, data_dev_, c);
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
  int one = 1;
  setToConstant(c);
  cublasStatus_t ret_cublas = cublasDdot(handle_cublas_, n_local_, ix.data_dev_, one, data_dev_, one, data_dev_);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
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
  int one = 1;
  copyFrom(vv);
  cublasStatus_t ret_cublas = cublasDdot(handle_cublas_, n_local_, ix.data_dev_, one, data_dev_, one, data_dev_);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
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

#if 0
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

void hiopVectorCuda::copy_to_two_vec_w_pattern(const hiopVector& c,
                                               const hiopVectorInt& c_map,
                                               const hiopVector& d,
                                               const hiopVectorInt& d_map)
{
  const int c_size = c.get_size();
  const int d_size = d.get_size();

  assert( c_size == c_map.size() );
  assert( d_size == d_map.size() );
  assert( c_size + d_size == n_local_);

  hiop::device::copy_mapped_src_to_dest_kernel(c_size, local_data_const(), c.local_data(), c_map.local_data_const());
  hiop::device::copy_mapped_src_to_dest_kernel(d_size, local_data_const(), d.local_data(), d_map.local_data_const());
}

#endif




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
  nrm = sqrt(nrmG);
#endif  
  return nrm;
}








#if 1
double hiopVectorCuda::onenorm_local() const
{
  // wrap raw pointer with a device_ptr 
  thrust_abs<double> abs_op;
  thrust::plus<double> plus_op;
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(data_dev_);
  
  // compute one norm
  double norm = thrust::transform_reduce(dev_v, dev_v+n_local_, abs_op, 0.0, plus_op);
  return norm;
}
#endif



void hiopVectorCuda::componentMult( const hiopVector& vec )
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec);
  assert(n_local_ == v.n_local_);

  // wrap raw pointer with a device_ptr 
  thrust::multiplies<double> thrust_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(data_dev_);
  thrust::device_ptr<double> dev_v2 = thrust::device_pointer_cast(v.data_dev_);
  
  thrust::transform(thrust::device,
                    dev_v1, dev_v1+n_local_,
                    dev_v2, dev_v2,
                    thrust_op);
}

void hiopVectorCuda::componentDiv( const hiopVector& vec )
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(vec);
  assert(n_local_ == v.n_local_);

  // wrap raw pointer with a device_ptr 
  thrust::divides<double> thrust_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(data_dev_);
  thrust::device_ptr<double> dev_v2 = thrust::device_pointer_cast(v.data_dev_);
  
  thrust::transform(thrust::device,
                    dev_v1, dev_v1+n_local_,
                    dev_v2, dev_v2,
                    thrust_op);
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
  int one = 1;
  axpy(alpha, xvec);
  cublasStatus_t ret_cublas = cublasDdot(handle_cublas_, n_local_, ix.data_dev_, one, data_dev_, one, data_dev_);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
}

}

