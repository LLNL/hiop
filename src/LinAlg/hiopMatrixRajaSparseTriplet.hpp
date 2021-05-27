// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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
 * @file hiopMatrixRajaSparseTriplet.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#pragma once

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixSparse.hpp"

#include <cassert>
#include <string>

namespace hiop
{

/** 
 * @brief Sparse matrix of doubles in triplet format - it is not distributed
 * @note for now (i,j) are expected ordered: first on rows 'i' and then on cols 'j'
 */
class hiopMatrixRajaSparseTriplet : public hiopMatrixSparse
{
public:
  hiopMatrixRajaSparseTriplet(int rows, int cols, int nnz, std::string memspace);
  virtual ~hiopMatrixRajaSparseTriplet(); 

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixSparse& dm);
  virtual void copy_to(int* irow, int* jcol, double* val);
  virtual void copy_to(hiopMatrixDense& W);

  virtual void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows);
  
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& Wmat, double alpha, const hiopMatrix& M2mat) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_);
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c) 
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /* block of W += alpha*transpose(this) */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						     double alpha, hiopMatrixDense& W) const;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							     double alpha, hiopMatrixDense& W) const
  {
    assert(false && "counterpart method of hiopMatrixRajaSymSparseTriplet should be used");
  }
  /* diag block of W += alpha * M * D^{-1} * transpose(M), where M=this 
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowCol_dest_start, const double& alpha, 
						       const hiopVector& D, hiopMatrixDense& W) const;

  /* block of W += alpha * M * D^{-1} * transpose(N), where M=this 
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and 
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the 
   * case, the method will assert(false) in debug; in release, the method will issue a 
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update 
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper 
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(int row_dest_start, int col_dest_start, const double& alpha,
					    const hiopVector& D, const hiopMatrixSparse& N,
					    hiopMatrixDense& W) const;

  virtual void copyRowsBlockFrom(const hiopMatrix& src_gen,
                                 const index_type& rows_src_idx_st,
                                 const size_type& n_rows,
                                 const index_type& rows_dest_idx_st,
                                 const size_type& dest_nnz_st)
  {
    assert(false && "not needed / implemented");
  }

  virtual void copySubmatrixFrom(const hiopMatrix& src_gen,
                                 const index_type& dest_row_st,
                                 const index_type& dest_col_st,
                                 const size_type& dest_nnz_st,
                                 const bool offdiag_only = false)
  {
    assert(false && "not needed / implemented");
  }
    
  virtual void copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                      const index_type& dest_row_st,
                                      const index_type& dest_col_st,
                                      const size_type& dest_nnz_st,
                                      const bool offdiag_only = false)
  {
    assert(false && "not needed / implemented");
  }  
  
  virtual void setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const int &nnz_to_copy,
                                                       const hiopVector& ix)
  {
    assert(false && "not needed / implemented");
  }

  virtual void setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const int &nnz_to_copy,
                                                       const hiopVector& ix)
  {
    assert(false && "not needed / implemented");
  }

  virtual void copyDiagMatrixToSubblock(const double& src_val,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const int &nnz_to_copy)
  {
    assert(false && "not needed / implemented");
  }

  virtual void copyDiagMatrixToSubblock_w_pattern(const hiopVector& x,
                                                  const index_type& dest_row_st,
                                                  const index_type& dest_col_st,
                                                  const size_type& dest_nnz_st,
                                                  const int &nnz_to_copy,
                                                  const hiopVector& pattern)
  {
    assert(false && "not needed / implemented");
  }
  
  virtual double max_abs_value();

  virtual void row_max_abs_value(hiopVector &ret_vec);
  
  virtual void scale_row(hiopVector &vec_scal, const bool inv_scale=false);

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start,
                                                    const double& alpha, 
                                                    hiopVector& vec_dest,
                                                    int vec_start,
                                                    int num_elems=-1) const 
  {
    assert(0 && "This method should be used only for symmetric matrices.\n");
  }

  virtual void convertToCSR(int &csr_nnz,
                            int **csr_kRowPtr,
                            int **csr_jCol,
                            double **csr_kVal,
                            int **index_covert_CSR2Triplet,
                            int **index_covert_extra_Diag2CSR)
  {
    assert(false && "not needed / implemented");
  }

  virtual size_type numberOfOffDiagNonzeros() const {assert("not implemented"&&0);return 0;};

  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS);

  virtual void set_Hess_FR(const hiopMatrixSparse& Hess,
                           int* iHSS,
                           int* jHSS,
                           double* MHSS,
                           const hiopVector& add_diag)
  {
    assert(false && "not needed / implemented");
  }

  virtual hiopMatrixSparse* alloc_clone() const;
  virtual hiopMatrixSparse* new_copy() const;

  inline int* i_row() { return iRow_; }
  inline int* j_col() { return jCol_; }
  inline double* M() { return values_; }

  inline const int* i_row() const { return iRow_; }
  inline const int* j_col() const { return jCol_; }
  inline const double* M() const { return values_; }
  
  inline int* i_row_host() { return iRow_host_; }
  inline int* j_col_host() { return jCol_host_; }
  inline double* M_host() { return values_host_; }

  inline const int* i_row_host() const { return iRow_host_; }
  inline const int* j_col_host() const { return jCol_host_; }
  inline const double* M_host() const { return values_host_; }

  void copyToDev();
  void copyFromDev() const;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return false; }
  virtual bool checkIndexesAreOrdered() const;
#endif
protected:
  int* iRow_; ///< row indices of the nonzero entries
  int* jCol_; ///< column indices of the nonzero entries
  double* values_; ///< values of the nonzero entries

  mutable int* iRow_host_;
  mutable int* jCol_host_;
  mutable double* values_host_;

  std::string mem_space_;// = "DEVICE";

protected:
  struct RowStartsInfo
  {
    int *idx_start_; //size num_rows+1
    int num_rows_;
    std::string mem_space_;
    RowStartsInfo()
      : idx_start_(NULL), num_rows_(0)
    {}
    RowStartsInfo(int n_rows, std::string memspace);
      //: idx_start(new int[n_rows+1]), num_rows(n_rows)
    virtual ~RowStartsInfo();
  };

  mutable RowStartsInfo* row_starts_host;
  //mutable RowStartsInfo* row_starts;

protected:
  RowStartsInfo* allocAndBuildRowStarts() const; 
private:
  hiopMatrixRajaSparseTriplet() 
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
  }
  hiopMatrixRajaSparseTriplet(const hiopMatrixRajaSparseTriplet&) 
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
    assert(false);
  }
};

/** Sparse symmetric matrix in triplet format. Only the upper triangle is stored */
class hiopMatrixRajaSymSparseTriplet : public hiopMatrixRajaSparseTriplet 
{
public: 
  hiopMatrixRajaSymSparseTriplet(int n, int nnz, std::string memspace)
    : hiopMatrixRajaSparseTriplet(n, n, nnz, memspace), nnz_offdiag_{-1}
  {
  }
  virtual ~hiopMatrixRajaSymSparseTriplet() {}  

  /** y = beta * y + alpha * this * x */
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const
  {
    return timesVec(beta, y, alpha, x);
  }
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const
  {
    return timesVec(beta, y, alpha, x);
  }

  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start,
						     double alpha, hiopMatrixDense& W) const;

  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
   							     double alpha, hiopMatrixDense& W) const;

  /* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
   * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
   * are available in 'vec_dest' starting at 'vec_start'
   */
  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start,
                                                    const double& alpha, 
                                                    hiopVector& vec_dest,
                                                    int vec_start, int num_elems=-1) const;
					    

  virtual hiopMatrixSparse* alloc_clone() const;
  virtual hiopMatrixSparse* new_copy() const;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return true; }
#endif
  virtual bool isDiagonal() const 
  {
    for(int itnnz = 0; itnnz < nnz_; itnnz++)
    {
      if(iRow_[itnnz] != jCol_[itnnz])
      {
        return false;
      }
    }
    return true;
  }
  
  virtual size_type numberOfOffDiagNonzeros() const;

  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS)
  {
    assert("not implemented"&&0);
  }
  
  virtual void set_Hess_FR(const hiopMatrixSparse& Hess,
                           int* iHSS,
                           int* jHSS,
                           double* MHSS,
                           const hiopVector& add_diag);

protected:
  mutable int nnz_offdiag_;     ///< number of nonzero entries

};
} //end of namespace
