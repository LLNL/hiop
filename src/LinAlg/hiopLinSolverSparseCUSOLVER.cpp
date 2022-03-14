//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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
 * @file hiopLinSolverSparseCUSOLVER.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 *
 */

#include "hiopLinSolverSparseCUSOLVER.hpp"

#include "hiop_blasdefs.hpp"

#include "klu.h"
#include "cusparse_v2.h"
#include <string>
#include <sstream>  


#define checkCudaErrors(val) hiopCheckCudaError((val), __FILE__, __LINE__)

namespace hiop
{
  hiopLinSolverIndefSparseCUSOLVER::hiopLinSolverIndefSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
  {
  }

  hiopLinSolverIndefSparseCUSOLVER::~hiopLinSolverIndefSparseCUSOLVER()
  {
    delete [] kRowPtr_;
    delete [] jCol_;
    delete [] kVal_;
    delete [] index_covert_CSR2Triplet_;
    delete [] index_covert_extra_Diag2CSR_;
    //KS: make sure we delete the GPU variables
    cudaFree(dia_);
    cudaFree(da_);
    cudaFree(dja_);
    cudaFree(devr_);
    cudaFree(devx_);
    cudaFree(d_work_);
    cusparseDestroy(handle_); 
    cusolverSpDestroy(handle_cusolver_);
    cublasDestroy(handle_cublas_);
    cusparseDestroyMatDescr(descr_A_);

    cusparseDestroyMatDescr(descr_M_);
    cusolverSpDestroyCsrluInfoHost(info_lu_);
    cusolverSpDestroyGluInfo(info_M_);

    klu_free_symbolic(&Symbolic_, &Common_) ;
    klu_free_numeric(&Numeric_, &Common_) ;
    free(mia_);
    free(mja_);
  }

  // Error checking utility for CUDA
  //KS: might later become part of src/Utils, putting it here for now

  template <typename T>
  void hiopLinSolverIndefSparseCUSOLVER::hiopCheckCudaError(T result,
                                                            const char *const file,
                                                            int const line)
  {
    if (result) {
      nlp_->log->printf(hovError, "CUDA error at %s:%d, error# %d\n", file, line, result);
      exit(EXIT_FAILURE);
    }
  }

  void hiopLinSolverIndefSparseCUSOLVER::firstCall()
  {

    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);

    kRowPtr_ = new int[n_+1]{0};

    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional dia_gonal elememts
    // the 1st part is sorted by row
    {
      //
      // compute nnz in each row
      //
      // off-dia_gonal part
      kRowPtr_[0] = 0;
      for(int k = 0; k < M_->numberOfNonzeros()-n_; k++) {
        if(M_->i_row()[k] != M_->j_col()[k]) {
          kRowPtr_[M_->i_row()[k]+1]++;
          kRowPtr_[M_->j_col()[k]+1]++;
          nnz_ += 2;
        }
      }
      // dia_gonal part
      for(int i=0;i<n_;i++) {
        kRowPtr_[i+1]++;
        nnz_ += 1;
      }
      // get correct row ptr index
      for(int i=1;i<n_+1;i++) {
        kRowPtr_[i] += kRowPtr_[i-1];
      }
      assert(nnz_ == kRowPtr_[n_]);

      kVal_ = new double[nnz_]{0.0};
      jCol_ = new int[nnz_]{0};

    }
    {
      //
      // set correct col index and value
      //
      index_covert_CSR2Triplet_ = new int[nnz_];
      index_covert_extra_Diag2CSR_ = new int[n_];

      int* nnz_each_row_tmp = new int[n_]{0};
      int total_nnz_tmp{0}, nnz_tmp{0}, rowID_tmp, colID_tmp;
      for(int k=0;k<n_;k++) {
        index_covert_extra_Diag2CSR_[k]=-1;
      }

      for(int k=0;k<M_->numberOfNonzeros()-n_;k++) {
        rowID_tmp = M_->i_row()[k];
        colID_tmp = M_->j_col()[k];
        if(rowID_tmp == colID_tmp) {
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          kVal_[nnz_tmp] += M_->M()[M_->numberOfNonzeros()-n_+rowID_tmp];
          index_covert_extra_Diag2CSR_[rowID_tmp] = nnz_tmp;

          nnz_each_row_tmp[rowID_tmp]++;
          total_nnz_tmp++;
        } else {
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_tmp = nnz_each_row_tmp[colID_tmp] + kRowPtr_[colID_tmp];
          jCol_[nnz_tmp] = rowID_tmp;
          kVal_[nnz_tmp] = M_->M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_each_row_tmp[rowID_tmp]++;
          nnz_each_row_tmp[colID_tmp]++;
          total_nnz_tmp += 2;
        }
      }
      // correct the missing dia_gonal term
      for(int i=0;i<n_;i++) {
        if(nnz_each_row_tmp[i] != kRowPtr_[i+1]-kRowPtr_[i]) {
          assert(nnz_each_row_tmp[i] == kRowPtr_[i+1]-kRowPtr_[i]-1);
          nnz_tmp = nnz_each_row_tmp[i] + kRowPtr_[i];
          jCol_[nnz_tmp] = i;
          kVal_[nnz_tmp] = M_->M()[M_->numberOfNonzeros()-n_+i];
          index_covert_CSR2Triplet_[nnz_tmp] = M_->numberOfNonzeros()-n_+i;
          total_nnz_tmp += 1;

          std::vector<int> ind_temp(kRowPtr_[i+1]-kRowPtr_[i]);
          std::iota(ind_temp.begin(), ind_temp.end(), 0);
          std::sort(ind_temp.begin(), ind_temp.end(),[&](int a, int b){ return jCol_[a+kRowPtr_[i]]<jCol_[b+kRowPtr_[i]]; });

          reorder(kVal_+kRowPtr_[i],ind_temp,kRowPtr_[i+1]-kRowPtr_[i]);
          reorder(index_covert_CSR2Triplet_+kRowPtr_[i],ind_temp,kRowPtr_[i+1]-kRowPtr_[i]);
          std::sort(jCol_+kRowPtr_[i],jCol_+kRowPtr_[i+1]);
        }
      }
      delete [] nnz_each_row_tmp;
    }


    /*
     * initialize KLU and cuSolver parameters
     */

    //handles
    cusparseCreate(&handle_); 
    cusolverSpCreate(&handle_cusolver_);
    cublasCreate(&handle_cublas_);

    //descriptors
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

    //info (data structure where factorization is stored)
    cusolverSpDestroyCsrluInfoHost(info_lu_);
    cusolverSpDestroyGluInfo(info_M_);
    cusolverSpCreateCsrluInfoHost(&info_lu_);
    cusolverSpCreateGluInfo(&info_M_);

    //KLU 
    klu_defaults(&Common_) ;

    // TODO: consider making this a part of setup options so that user can
    // set up these values. For now, we keep them hard-wired. 
    Common_.btf = 0;
    Common_.ordering = 1; //COLAMD; use 0 for AMD
    Common_.tol = 0.1;
    Common_.scale = -1;
    Common_.halt_if_singular=0;
    //allocate gpu data
    free(mia_); 
    free(mja_);
    mia_ = nullptr; 
    mja_ = nullptr;
    cudaFree(devx_);
    cudaFree(devr_);
    devx_ = nullptr;
    devr_ = nullptr;
    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));
    this->newKLUfactorization();
  }

  //helper private function needed for format conversion

  int hiopLinSolverIndefSparseCUSOLVER::createM(const int n, 
                                                const int /* nnzL */, 
                                                const int* Lp, 
                                                const int* Li,
                                                const int /* nnzU */, 
                                                const int* Up, 
                                                const int* Ui){

    int row;
    for(int i = 0;i<n;++i) {
      //go through EACH COLUMN OF L first
      for(int j = Lp[i]; j<Lp[i+1]; ++j) {
        row = Li[j];
        //BUT dont count dia_gonal twice, important
        if(row != i) {
          mia_[row + 1]++;
        }
      }
      //then each column of U
      for(int j = Up[i];j<Up[i+1];++j) {
        row = Ui[j];
        mia_[row + 1]++;
      }
    }
    //then organize mia_;
    mia_[0] = 0;
    for(int i=1;i<n+1;i++) {
      mia_[i] += mia_[i-1];
    } 

    std::vector<int> Mshifts(n, 0);
    for(int i = 0;i<n;++i) {

      //go through EACH COLUMN OF L first
      for(int j = Lp[i];j<Lp[i+1];++j) {
        row = Li[j];
        if(row != i) {
          //place (row, i) where it belongs!

          mja_[mia_[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      //each column of U next
      for(int j = Up[i];j<Up[i+1];++j) {
        row = Ui[j];
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    return 0;
  }


  // call if both the matrix and the nnz structure changed or if convergence is poor while using refactorization.
  int hiopLinSolverIndefSparseCUSOLVER::newKLUfactorization()
  {
    klu_free_symbolic(&Symbolic_, &Common_) ;
    klu_free_numeric(&Numeric_, &Common_) ;
    Symbolic_ = klu_analyze(n_, kRowPtr_, jCol_, &Common_) ;
    if(Symbolic_ == nullptr) {
      nlp_->log->printf(hovError, "symbolic nullptr\n");
      return -1;
    }

    Numeric_ = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic_, &Common_);
    if(Numeric_ == nullptr) {
      nlp_->log->printf(hovError, "printf matrix size: %d numeric nullptr \n", n_);
      return -1;
    }
    // get sizes
    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;

    const int nnzM = (nnzL + nnzU - n_);

    /* parse the factorization */

    free(mia_);
    free(mja_);

    mia_ = (int*) calloc(sizeof(int), (n_+1));
    mja_ = (int*) calloc(sizeof(int), nnzM);
    int* Lp = new int[n_+1];
    int* Li = new int[nnzL];
    //we cant use nullptr instrad od Lx and Ux because it causes SEG FAULT. It seems like a waste of memory though.
    double* Lx = new double[nnzL];
    int* Up = new int[n_+1];
    int* Ui = new int[nnzU];
    double* Ux = new double[nnzU];
    int ok = klu_extract(Numeric_, Symbolic_, Lp, Li, Lx, Up, Ui, Ux, 
                         nullptr, nullptr, nullptr, 
                         nullptr, nullptr, 
                         nullptr, nullptr, &Common_);
    createM(n_, nnzL, Lp, Li,nnzU, Up, Ui);
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;

    /* setup GLU */ 
    sp_status_ = cusolverSpDgluSetup(handle_cusolver_,
                                     n_,
                                     nnz_,
                                     descr_A_,
                                     kRowPtr_,
                                     jCol_,
                                     Numeric_->Pnum, /* base-0 */
                                     Symbolic_->Q, /* base-0 */
                                     nnzM, /* nnzM */
                                     descr_M_,
                                     mia_,
                                     mja_,
                                     info_M_);

    sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_,
                                          info_M_,
                                          &size_M_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    buffer_size_ = size_M_;
    cudaFree(d_work_);
    cudaMalloc((void **)&d_work_, buffer_size_);
    sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_,
                                        info_M_,
                                        d_work_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    //now make sure the space is allocated for A on the GPU (but dont copy)
    cudaFree(da_); 
    cudaFree(da_);
    cudaFree(dia_);
    checkCudaErrors(cudaMalloc(&da_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia_, (n_ +1)* sizeof(int)));
    //copy    

    checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia_, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja_, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
    //reset and refactor so factors are ON THE GPU

    sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     da_,
                                     dia_,
                                     dja_,
                                     info_M_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    sp_status_ = cusolverSpDgluFactor(handle_cusolver_,
                                      info_M_,
                                      d_work_);
    return 0;
  }

  int hiopLinSolverIndefSparseCUSOLVER::matrixChanged()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if(!kRowPtr_){
      this->firstCall();
    } else {
      // update matrix
      for(int k=0;k<nnz_;k++) {
        kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i=0;i<n_;i++) {
        if(index_covert_extra_Diag2CSR_[i] != -1)
          kVal_[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros()-n_+i];
      }
      // somehow update the matrix not sure how

      // call new factorization if necessary
      // update the GPU matrix

      checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
      //re-factor here

      sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                       n_,
                                       /* A is original matrix */
                                       nnz_,
                                       descr_A_,
                                       da_,
                                       dia_,
                                       dja_,
                                       info_M_);

      sp_status_ = cusolverSpDgluFactor(handle_cusolver_,
                                        info_M_,
                                        d_work_);
      //end of factor
    }    
    return 0;
  }

  bool hiopLinSolverIndefSparseCUSOLVER::solve ( hiopVector& x )
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);
    assert(x.get_size() == M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVector* rhs = x.new_copy();
    double* dx = x.local_data();
    double* drhs = rhs->local_data();

    checkCudaErrors(cudaMemcpy(devr_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    //solve HERE

    sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     da_,
                                     dia_,
                                     dja_,
                                     devr_, /* right hand side */
                                     devx_,   /* left hand side */
                                     &ite_refine_succ_,
                                     &r_nrminf_,
                                     info_M_,
                                     d_work_);
    //copy the solutuion back: dx = devx_
    if(sp_status_ == 0){
      checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();

    delete rhs; rhs = nullptr;
    return 1;
  }

  //
  // The Solver for Nonsymmetric KKT System
  //

  hiopLinSolverNonSymSparseCUSOLVER::hiopLinSolverNonSymSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
  {}

  hiopLinSolverNonSymSparseCUSOLVER::~hiopLinSolverNonSymSparseCUSOLVER()
  {
    delete [] kRowPtr_;
    delete [] jCol_;
    delete [] kVal_;
    delete [] index_covert_CSR2Triplet_;
    delete [] index_covert_extra_Diag2CSR_;
    //KS: make sure we delete the GPU variables
    cudaFree(dia_);
    cudaFree(da_);
    cudaFree(dja_);
    cudaFree(devr_);
    cudaFree(devx_);
    cudaFree(d_work_);
    cusparseDestroy(handle_); 
    cusolverSpDestroy(handle_cusolver_);
    cublasDestroy(handle_cublas_);
    cusparseDestroyMatDescr(descr_A_);

    cusparseDestroyMatDescr(descr_M_);
    cusolverSpDestroyCsrluInfoHost(info_lu_);
    cusolverSpDestroyGluInfo(info_M_);

    klu_free_symbolic(&Symbolic_, &Common_) ;
    klu_free_numeric(&Numeric_, &Common_) ;
    free(mia_);
    free(mja_);
  }

  int hiopLinSolverNonSymSparseCUSOLVER::createM(const int n, 
                                                 const int /* nnzL */, 
                                                 const int* Lp, 
                                                 const int* Li,
                                                 const int /* nnzU */, 
                                                 const int* Up, 
                                                 const int* Ui)
  {
    int row;
    for(int i = 0; i<n; ++i) {
      //go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i+1]; ++j) {
        row = Li[j];
        //BUT dont count dia_gonal twice, important
        if(row != i){
          mia_[row+1]++;
        }
      }
      //then each column of U
      for(int j = Up[i]; j < Up[i+1]; ++j) {
        row = Ui[j];
        mia_[row+1]++;
      }
    }
    //then organize mia_;
    mia_[0] = 0;
    for(int i = 1; i < n+1; ++i) {
      mia_[i] += mia_[i - 1];
    } 

    int* Mshifts = (int*) calloc (n, sizeof(int));
    for(int i = 0; i < n; ++i) {

      //go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i+1]; ++j) {
        row = Li[j];
        if(row != i){
          //place (row, i) where it belongs!
          mja_[mia_[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      //each column of U next
      for(int j = Up[i]; j < Up[i+1]; ++j) {
        row = Ui[j];
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    free(Mshifts);
    return 0;
  }

  template <typename T>
  void hiopLinSolverNonSymSparseCUSOLVER::hiopCheckCudaError(T result,
                                                             const char *const file,
                                                             int const line)
  {
    if (result) {
      nlp_->log->printf(hovError, "CUDA error at %s:%d, error# %d\n", file, line, result);
      exit(EXIT_FAILURE);
    }
  }

  void hiopLinSolverNonSymSparseCUSOLVER::firstCall()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);

    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional dia_gonal elememts
    // the 1st part is sorted by row
    hiop::hiopMatrixSparseTriplet* M_triplet = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(M_);
    if(M_triplet == nullptr) {
      nlp_->log->printf(hovError, "M_triplet is nullptr");
      return;
    }

    M_triplet->convertToCSR(nnz_, &kRowPtr_, &jCol_, &kVal_, &index_covert_CSR2Triplet_, &index_covert_extra_Diag2CSR_, extra_dia_g_nnz_map);

    /*
     * initialize cusolver parameters
     */

    //handles
    cusparseCreate(&handle_); 
    cusolverSpCreate(&handle_cusolver_);
    cublasCreate(&handle_cublas_);

    //descriptors
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

    //info (data structure where factorization is stored)
    cusolverSpCreateCsrluInfoHost(&info_lu_);
    cusolverSpCreateGluInfo(&info_M_);

    //KLU 

    klu_defaults(&Common_) ;

    // TODO: consider making a part of setup options that can be called from a user side
    // For now, keeping these options hard-wired
    Common_.btf = 0;
    Common_.ordering = 1;//COLAMD; use 0 for AMD
    Common_.tol = 0.1;
    Common_.scale = -1;
    Common_.halt_if_singular=0;

    free(mia_); 
    free(mja_);
    mia_ = nullptr;
    mja_ = nullptr;
    //allocate gpu data
    cudaFree(devx_);
    cudaFree(devr_);
    devx_ = nullptr;
    devr_ = nullptr;
    checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));
    this->newKLUfactorization();
  }


  int hiopLinSolverNonSymSparseCUSOLVER::newKLUfactorization()
  {

    Symbolic_ = klu_analyze(n_, kRowPtr_, jCol_, &Common_) ;

    if(Symbolic_ == nullptr){
      nlp_->log->printf(hovError, "symbolic nullptr");
      return -1;
    }

    Numeric_ = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic_, &Common_);
    if(Numeric_ == nullptr){
      nlp_->log->printf(hovError, "numeric  nullptr");
      return -1;
    }


    // get sizes

    const int nnzL = Numeric_->lnz;
    const int nnzU = Numeric_->unz;

    const int nnzM = (nnzL+nnzU-n_);
    /* parse the factorization */

    free(mia_);
    free(mja_);

    mia_ = (int*) calloc(sizeof(int), (n_+1));
    mja_ = (int*) calloc(sizeof(int), nnzM);

    int* Lp = new int[n_+1];
    int* Li = new int[nnzL];
    //we cant use nullptr instrad od Lx and Ux because it causes SEG FAULT. It seems like a waste of memory though.
    double* Lx = new double[nnzL];
    int* Up = new int[n_+1];
    int* Ui = new int[nnzU];
    double* Ux = new double[nnzU];
    int ok = klu_extract(Numeric_, 
                         Symbolic_,
                         Lp,
                         Li,
                         Lx,
                         Up,
                         Ui,
                         Ux, 
                         nullptr,
                         nullptr,
                         nullptr, 
                         nullptr,
                         nullptr, 
                         nullptr,
                         nullptr,
                         &Common_);
    createM(n_, nnzL, Lp, Li, nnzU, Up, Ui);
    delete[] Lp;
    delete[] Li;
    delete[] Lx;
    delete[] Up;
    delete[] Ui;
    delete[] Ux;

    /* setup GLU */ 
    sp_status_ = cusolverSpDgluSetup(handle_cusolver_,
                                     n_,
                                     nnz_,
                                     descr_A_,
                                     kRowPtr_,
                                     jCol_,
                                     Numeric_->Pnum, /* base-0 */
                                     Symbolic_->Q, /* base-0 */
                                     nnzM, /* nnzM */
                                     descr_M_,
                                     mia_,
                                     mja_,
                                     info_M_);

    sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_,
                                          info_M_,
                                          &size_M_);

    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
    buffer_size_ = size_M_;
    cudaFree(d_work_);
    checkCudaErrors(cudaMalloc((void **)&d_work_, buffer_size_));
    sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_,
                                        info_M_,
                                        d_work_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    //now make sure the space is allocated for A on the GPU (but dont copy)
    cudaFree(da_); 
    cudaFree(da_);
    cudaFree(dia_);
    checkCudaErrors(cudaMalloc(&da_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia_, (n_ +1)* sizeof(int)));
    //dont free d_ia -> size is n+1 so doesnt matter
    if(dia_ == nullptr) {
      checkCudaErrors(cudaMalloc(&dia_, (n_+1) * sizeof(int)));
    }
    checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia_, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja_, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));


    //reset and refactor so factors are ON THE GPU

    sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     da_,
                                     dia_,
                                     dja_,
                                     info_M_);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

    return 0;

  }

  int hiopLinSolverNonSymSparseCUSOLVER::matrixChanged()
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if(!kRowPtr_) {
      this->firstCall();
    } else {
      // update matrix
      for(int k=0;k<nnz_;k++) {
        kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i=0;i<n_;i++) {
        if(index_covert_extra_Diag2CSR_[i] != -1) {
          kVal_[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
        }     
      }

      checkCudaErrors(cudaMemcpy(da_, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));

      //re-factor here

      sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                       n_,
                                       /* A is original matrix */
                                       nnz_,
                                       descr_A_,
                                       da_,
                                       dia_,
                                       dja_,
                                       info_M_);
      //end of factor
    }

    return 0;
  }

  bool hiopLinSolverNonSymSparseCUSOLVER::solve(hiopVector& x_)
  {
    assert(n_ == M_->n() && M_->n() == M_->m());
    assert(n_>0);
    assert(x_.get_size() == M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != nullptr);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    x->copyToDev();
    double* dx = x->local_data();
    //rhs->copyToDev();
    double* drhs = rhs->local_data();
    // double* devr_ = rhs->local_data();
    checkCudaErrors(cudaMemcpy(devr_, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    //solve HERE

    sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     da_,
                                     dia_,
                                     dja_,
                                     devr_, /* right hand side */
                                     devx_,   /* left hand side */
                                     &ite_refine_succ_,
                                     &r_nrminf_,
                                     info_M_,
                                     d_work_);
    //copy the solutuion back

    checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    // set dx = devx_; 
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    delete rhs; rhs = nullptr;
    return 1;
  }

} //namespace hiop
