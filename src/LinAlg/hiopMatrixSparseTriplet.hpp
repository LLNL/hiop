#ifndef HIOP_SPARSE_MATRIX_TRIPLET
#define HIOP_SPARSE_MATRIX_TRIPLET

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixSparse.hpp"

#include <cassert>
#include <unordered_map>

namespace hiop
{

/**
 * @brief Sparse matrix of doubles in triplet format - it is not distributed
 * @note for now (i,j) are expected ordered: first on rows 'i' and then on cols 'j'
 *
 * Note: the following methods of hiopMatrix are NOT
 * implemented in this class:
 * - addSubDiagonal
 * - addUpperTriangleToSymDenseMatrixUpperTriangle
 * - startingAtAddSubDiagonalToStartingAt
 */
class hiopMatrixSparseTriplet : public hiopMatrixSparse
{
public:
  hiopMatrixSparseTriplet(int rows, int cols, int nnz);
  virtual ~hiopMatrixSparseTriplet();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixSparse& dm);

  virtual void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows);

  virtual void timesVec(double beta,  hiopVector& y, double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y, double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y, double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta,   double* y, double alpha, const double* x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_);
  
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha,
			      const hiopVector& d_,
                              int start_on_src_vec,
                              int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
  * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems', scaled by 'scal'
  */
  virtual void copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                   const size_type& num_elems,
                                   const hiopVector& d_,
                                   const index_type& start_on_nnz_idx,
                                   double scal=1.0);

  /* add constant 'c' to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements.
  * The number of elements added is 'num_elems'
  */
  virtual void setSubDiagonalTo(const index_type& start_on_dest_diag,
                                const size_type& num_elems,
                                const double& c,
                                const index_type& start_on_nnz_idx);

  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /* block of W += alpha*transpose(this), where W is dense */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
						     double alpha,
                                                     hiopMatrixDense& W) const;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start,
							     double alpha,
                                                             hiopMatrixDense& W) const
  {
    assert(false && "counterpart method of hiopMatrixSymSparseTriplet should be used");
  }

  virtual void addUpperTriangleToSymSparseMatrixUpperTriangle(int diag_start,
                                                              double alpha,
                                                              hiopMatrixSparse& W) const
  {
    assert(false && "implemented only for symmetric matrices");
  }

  /* diag block of W += alpha * M * D^{-1} * transpose(M), where M=this
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowCol_dest_start,
                                                       const double& alpha,
						       const hiopVector& D,
                                                       hiopMatrixDense& W) const;

  /* block of W += alpha * M * D^{-1} * transpose(N), where M=this
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the
   * case, the method will assert(false) in debug; in release, the method will issue a
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(int row_dest_start,
                                            int col_dest_start,
                                            const double& alpha,
                                            const hiopVector& D,
                                            const hiopMatrixSparse& N,
                                            hiopMatrixDense& W) const;

  virtual void copyRowsBlockFrom(const hiopMatrix& src_gen,
                                 const index_type& rows_src_idx_st,
                                 const size_type& n_rows,
                                 const index_type& rows_dest_idx_st,
                                 const size_type& dest_nnz_st);

  /**
  * @brief Copy matrix 'src_gen', into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
  * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements. 
  * When `offdiag_only` is set to true, only the off-diagonal part of `src_gen` is copied.
  *
  * @pre 'this' must have enough rows and cols after row 'dest_row_st' and col 'dest_col_st'
  * @pre 'dest_nnz_st' + the number of non-zeros in the copied matrix must be less or equal to 
  * this->numOfNumbers()
  * @pre User must know the nonzero pattern of src and dest matrices. The method assumes 
  * that non-zero patterns does not change between calls and that 'src_gen' is a valid
  *  submatrix of 'this'
  */
  virtual void copySubmatrixFrom(const hiopMatrix& src_gen,
                                 const index_type& dest_row_st,
                                 const index_type& dest_col_st,
                                 const size_type& dest_nnz_st,
                                 const bool offdiag_only = false);
  
  /**
  * @brief Copy the transpose of matrix 'src_gen', into 'this' as a submatrix from corner 
  * 'dest_row_st' and 'dest_col_st'.
  * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
  * When `offdiag_only` is set to true, only the off-diagonal part of `src_gen` is copied.
  */
  virtual void copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                      const index_type& dest_row_st,
                                      const index_type& dest_col_st,
                                      const size_type& dest_nnz_st,
                                      const bool offdiag_only = false);

  /**
  * @brief Copy the selected cols/rows of a diagonal matrix (a constant 'scalar' times identity),
  * into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
  * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
  */
  virtual void setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const int &nnz_to_copy,
                                                       const hiopVector& ix);
  virtual void setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const int &nnz_to_copy,
                                                       const hiopVector& ix);

  /**
  * @brief Copy a diagonal matrix to destination.
  * This diagonal matrix is 'src_val'*identity matrix with size 'n_rows'x'n_rows'.
  * The destination is updated from the start row 'row_dest_st' and start column 'col_dest_st'.
  */
  virtual void copyDiagMatrixToSubblock(const double& src_val,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const int &nnz_to_copy);
  /** 
   * @brief same as @copyDiagMatrixToSubblock, but copies only diagonal entries specified by 'pattern' 
   */
  virtual void copyDiagMatrixToSubblock_w_pattern(const hiopVector& x,
                                                  const index_type& dest_row_st,
                                                  const index_type& dest_col_st,
                                                  const size_type& dest_nnz_st,
                                                  const size_type &nnz_to_copy,
                                                  const hiopVector& pattern);

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
                            int **index_covert_extra_Diag2CSR,
                            std::unordered_map<int,int> &extra_diag_nnz_map);

  virtual size_type numberOfOffDiagNonzeros() const {assert("not implemented"&&0);return 0;};

  /// @brief extend base problem Jac to the Jac in feasibility problem
  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS);

  /// @brief extend base problem Hess to the Hess in feasibility problem
  virtual void set_Hess_FR(const hiopMatrixSparse& Hess,
                           int* iHSS,
                           int* jHSS,
                           double* MHSS,
                           const hiopVector& add_diag) {assert("not implemented"&&0);}

  virtual hiopMatrixSparse* alloc_clone() const;
  virtual hiopMatrixSparse* new_copy() const;

  inline int* i_row() { return iRow_; }
  inline int* j_col() { return jCol_; }
  inline double* M() { return values_; }

  inline const int* i_row() const { return iRow_; }
  inline const int* j_col() const { return jCol_; }
  inline const double* M() const { return values_; }


#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return false; }
  virtual bool checkIndexesAreOrdered() const;
#endif
protected:
  int* iRow_; ///< row indices of the nonzero entries
  int* jCol_; ///< column indices of the nonzero entries
  double* values_; ///< values_ of the nonzero entries

protected:
  struct RowStartsInfo
  {
    int *idx_start_; //size num_rows+1
    int num_rows_;
    RowStartsInfo()
      : idx_start_(NULL), num_rows_(0)
    {}
    RowStartsInfo(int n_rows)
      : idx_start_(new int[n_rows+1]), num_rows_(n_rows)
    {}
    virtual ~RowStartsInfo()
    {
      delete[] idx_start_;
    }
  };
  mutable RowStartsInfo* row_starts_;
protected:
  RowStartsInfo* allocAndBuildRowStarts() const;
private:
  hiopMatrixSparseTriplet()
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
  }
  hiopMatrixSparseTriplet(const hiopMatrixSparseTriplet&)
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
    assert(false);
  }
};

/** Sparse symmetric matrix in triplet format. Only the upper triangle is stored */
class hiopMatrixSymSparseTriplet : public hiopMatrixSparseTriplet
{
public:
  hiopMatrixSymSparseTriplet(int n, int nnz)
    : hiopMatrixSparseTriplet(n, n, nnz), nnz_offdiag_{-1}
  {}
  virtual ~hiopMatrixSymSparseTriplet() {}

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
  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start, const double& alpha,
					    hiopVector& vec_dest, int vec_start, int num_elems=-1) const;

  virtual hiopMatrixSparse* alloc_clone() const;
  virtual hiopMatrixSparse* new_copy() const;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return true; }
#endif
  virtual bool isDiagonal() const
  {
    for(int itnnz=0; itnnz<nnz_; itnnz++) if(iRow_[itnnz]!=jCol_[itnnz]) return false;
    return true;
  }

  virtual size_type numberOfOffDiagNonzeros() const;

  /// @brief extend base problem Jac to the Jac in feasibility problem
  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS){assert("not implemented"&&0);};

  /// @brief extend base problem Hess to the Hess in feasibility problem
  virtual void set_Hess_FR(const hiopMatrixSparse& Hess,
                           int* iHSS,
                           int* jHSS,
                           double* MHSS,
                           const hiopVector& add_diag);

protected:
  mutable int nnz_offdiag_;     ///< number of nonzero entries

};

} //end of namespace

#endif
