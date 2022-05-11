// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
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
 * @file hiopLinSolverSparseGinkgo.cpp
 *
 * @author Fritz Goebel <fritz.goebel@kit.edu>, KIT
 *
 */

#include "hiopLinSolverSparseGinkgo.hpp"

#include "hiop_blasdefs.hpp"


namespace hiop
{


namespace
{


std::shared_ptr<gko::matrix::Csr<double, int>> transferTripletToCSR(std::shared_ptr<gko::Executor> exec,
                                                                    int n_,
                                                                    hiopMatrixSparse* M_,
                                                                    int** index_covert_CSR2Triplet,
                                                                    int** index_covert_extra_Diag2CSR)
{
    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional diagonal elememts
    // the 1st part is sorted by row
    int nnz_{0};
    auto kRowPtr_ = new int[n_+1]{0};
    {
      //
      // compute nnz in each row
      //
      // off-diagonal part
      kRowPtr_[0]=0;
      for(int k=0; k<M_->numberOfNonzeros()-n_; k++) {
        if(M_->i_row()[k]!=M_->j_col()[k]) {
          kRowPtr_[M_->i_row()[k]+1]++;
          kRowPtr_[M_->j_col()[k]+1]++;
          nnz_ += 2;
        }
      }
      // diagonal part
      for(int i=0; i<n_; i++) {
        kRowPtr_[i+1]++;
        nnz_ += 1;
      }
      // get correct row ptr index
      for(int i=1; i<n_+1; i++) {
        kRowPtr_[i] += kRowPtr_[i-1];
      }
      assert(nnz_==kRowPtr_[n_]);
    }
    auto kVal_ = new double[nnz_]{0.0};
    auto jCol_ = new int[nnz_]{0};
    *index_covert_CSR2Triplet = new int[nnz_];
    *index_covert_extra_Diag2CSR = new int[n_];
    auto index_covert_CSR2Triplet_ = *index_covert_CSR2Triplet;
    auto index_covert_extra_Diag2CSR_ = *index_covert_extra_Diag2CSR;
    {
      //
      // set correct col index and value
      //

      int *nnz_each_row_tmp = new int[n_]{0};
      int total_nnz_tmp{0};
      int nnz_tmp{0};
      int rowID_tmp;
      int colID_tmp;
      for(int k=0; k<n_; k++) {
        index_covert_extra_Diag2CSR_[k] = -1;
      }

      for(int k=0; k<M_->numberOfNonzeros()-n_; k++) {
        rowID_tmp = M_->i_row()[k];
        colID_tmp = M_->j_col()[k];
        if(rowID_tmp==colID_tmp) {
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
      // correct the missing diagonal term
      for(int i=0; i<n_; i++) {
        if(nnz_each_row_tmp[i] != kRowPtr_[i+1] - kRowPtr_[i]) {
          assert(nnz_each_row_tmp[i] == kRowPtr_[i+1] - kRowPtr_[i] - 1);
          nnz_tmp = nnz_each_row_tmp[i] + kRowPtr_[i];
          jCol_[nnz_tmp] = i;
          kVal_[nnz_tmp] = M_->M()[M_->numberOfNonzeros() - n_ + i];
          index_covert_CSR2Triplet_[nnz_tmp] = M_->numberOfNonzeros() - n_ + i;
          total_nnz_tmp += 1;

          std::vector<int> ind_temp(kRowPtr_[i+1] - kRowPtr_[i]);
          std::iota(ind_temp.begin(), ind_temp.end(), 0);
          std::sort(ind_temp.begin(), ind_temp.end(),[&](int a, int b){ return jCol_[a+kRowPtr_[i]] < jCol_[b+kRowPtr_[i]]; });

          reorder(kVal_+kRowPtr_[i],ind_temp,kRowPtr_[i+1] - kRowPtr_[i]);
          reorder(index_covert_CSR2Triplet_+kRowPtr_[i],ind_temp,kRowPtr_[i+1] - kRowPtr_[i]);
          std::sort(jCol_+kRowPtr_[i],jCol_+kRowPtr_[i+1]);
        }
      }

      delete[] nnz_each_row_tmp;
    }

    auto val_array = gko::Array<double>::view(exec, nnz_, kVal_);
    auto row_ptrs = gko::Array<int>::view(exec, n_ + 1, kRowPtr_);
    auto col_idxs = gko::Array<int>::view(exec, nnz_, jCol_);
    auto mtx = gko::share(gko::matrix::Csr<double, int>::create(exec, gko::dim<2>{n_, n_}, val_array, col_idxs, row_ptrs));
    return mtx;
}


void update_matrix(hiopMatrixSparse* M_,
                   std::shared_ptr<gko::matrix::Csr<double, int>> mtx,
                   int* index_covert_CSR2Triplet_,
                   int* index_covert_extra_Diag2CSR_)
{
    int n_ = mtx->get_size()[0];
    int nnz_= mtx->get_num_stored_elements();
    auto values = mtx->get_values();
    int rowID_tmp{0};
    for(int k=0; k<nnz_; k++) {
        values[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(int i=0; i<n_; i++) {
        if(index_covert_extra_Diag2CSR_[i] != -1) {
            values[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
        }
    }
}


std::shared_ptr<gko::LinOpFactory> setup_solver_factory(std::shared_ptr<const gko::Executor> exec,
                                                        std::shared_ptr<gko::matrix::Csr<double, int>> mtx)
{

    auto preprocessing_fact = gko::share(gko::reorder::Mc64<double, int>::build().on(exec));
    auto preprocessing = gko::share(preprocessing_fact->generate(mtx));

    auto lu_fact = gko::share(gko::factorization::Glu<double, int>::build_reusable()
                              .on(exec, mtx.get(), preprocessing.get()));
    auto inner_solver_fact = gko::share(gko::preconditioner::Ilu<>::build()
                                        .with_factorization_factory(lu_fact)
                                        .on(exec));
    auto solver_fact = gko::share(gko::solver::Gmres<>::build()
                                  .with_criteria(
                                    gko::stop::Iteration::build()
                                      .with_max_iters(200u)
                                      .on(exec),
                                    gko::stop::ResidualNorm<>::build()
                                      .with_baseline(gko::stop::mode::absolute)
                                      .with_reduction_factor(1e-8)
                                      .on(exec))
                                  .with_krylov_dim(10u)
                                  .with_preconditioner(inner_solver_fact)
                                  .on(exec));

    auto reusable_factory = gko::share(gko::solver::ScaledReordered<>::build()
                                       .with_solver(solver_fact)
                                       .with_reordering(preprocessing)
                                       .on(exec));
    return reusable_factory;
}


}


  hiopLinSolverSymSparseGinkgo::hiopLinSolverSymSparseGinkgo(const int& n, 
                                                             const int& nnz,
                                                             hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
  {}

  hiopLinSolverSymSparseGinkgo::~hiopLinSolverSymSparseGinkgo()
  {
    delete [] index_covert_CSR2Triplet_;
    delete [] index_covert_extra_Diag2CSR_;
  }

  void hiopLinSolverSymSparseGinkgo::firstCall()
  {
    nlp_->log->printf(hovSummary, "Setting up Ginkgo solver ... \n");
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);

    exec_ = gko::ReferenceExecutor::create(); //gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    mtx_ = transferTripletToCSR(exec_, n_, M_, &index_covert_CSR2Triplet_, &index_covert_extra_Diag2CSR_);
    nnz_ = mtx_->get_num_stored_elements();

    reusable_factory_ = setup_solver_factory(exec_, mtx_);
  }

  int hiopLinSolverSymSparseGinkgo::matrixChanged()
  {
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if( !mtx_ ) {
      this->firstCall();
    } else {
      update_matrix(M_, mtx_, index_covert_CSR2Triplet_, index_covert_extra_Diag2CSR_);
    }

    gko_solver_ = gko::share(reusable_factory_->generate(mtx_));
    return 0; // This needs to be changed to return -1 if the matrix is singular - as soon as ginkgo supports this.
  }

  bool hiopLinSolverSymSparseGinkgo::solve ( hiopVector& x_ )
  {
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);
    assert(x_.get_size()==M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    double* dx = x->local_data();
    double* drhs = rhs->local_data();
    auto x_array = gko::Array<double>::view(exec_, n_, dx);
    auto b_array = gko::Array<double>::view(exec_, n_, drhs);
    auto dense_x = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{n_, 1}, gko::Array<double>::view(exec_, n_, dx), 1);
    auto dense_b = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{n_, 1}, b_array, 1);

    gko_solver_->apply(dense_b.get(), dense_x.get());

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    
    delete rhs; rhs=nullptr;
    return 1;
  }


  hiopLinSolverNonSymSparseGinkgo::hiopLinSolverNonSymSparseGinkgo(const int& n,
                                                                   const int& nnz,
                                                                   hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
  {}

  hiopLinSolverNonSymSparseGinkgo::~hiopLinSolverNonSymSparseGinkgo()
  {
    if(index_covert_CSR2Triplet_) {
      delete [] index_covert_CSR2Triplet_;
    }
    if(index_covert_extra_Diag2CSR_) {
      delete [] index_covert_extra_Diag2CSR_;
    }
  }
  
  void hiopLinSolverNonSymSparseGinkgo::firstCall()
  {
    nlp_->log->printf(hovSummary, "Setting up Ginkgo solver ... \n");
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);

    exec_= gko::ReferenceExecutor::create(); //gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    mtx_ = transferTripletToCSR(exec_, n_, M_, &index_covert_CSR2Triplet_, &index_covert_extra_Diag2CSR_);
    nnz_ = mtx_->get_num_stored_elements();

    reusable_factory_ = setup_solver_factory(exec_, mtx_);
  }

  int hiopLinSolverNonSymSparseGinkgo::matrixChanged()
  {
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if( !mtx_ ) {
      this->firstCall();
    } else {
      update_matrix(M_, mtx_, index_covert_CSR2Triplet_, index_covert_extra_Diag2CSR_);
    }

    gko_solver_ = gko::share(reusable_factory_->generate(mtx_));

    return 0; // This needs to be changed to return -1 if the matrix is singular - as soon as ginkgo supports this.
  }

  bool hiopLinSolverNonSymSparseGinkgo::solve(hiopVector& x_)
  {
    assert(n_==M_->n() && M_->n()==M_->m());
    assert(n_>0);
    assert(x_.get_size()==M_->n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    double* dx = x->local_data();
    double* drhs = rhs->local_data();
    auto x_array = gko::Array<double>::view(exec_, n_, dx);
    auto b_array = gko::Array<double>::view(exec_, n_, drhs);
    auto dense_x = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{n_, 1}, x_array, 1);
    auto dense_b = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{n_, 1}, b_array, 1);

    gko_solver_->apply(dense_b.get(), dense_x.get());

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    
    delete rhs; rhs=nullptr;
    return 1;
  }


} //end namespace hiop
