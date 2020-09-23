#pragma once

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"

#include <cassert>

namespace hiop
{

class hiopMatrixSparseTriplet;
/**
 * @brief Sparse matrix of doubles in triplet format - it is not distributed
 * @note for now (i,j) are expected ordered: first on rows 'i' and then on cols 'j'
 */
class hiopMatrixSparse : public hiopMatrix
{
public:
  hiopMatrixSparse(int rows, int cols, int nnz)
      : nrows_(rows)
      , ncols_(cols)
      , nnz_(nnz)
  {
  }
  virtual ~hiopMatrixSparse()
  {
  }

  virtual void setToZero() = 0;
  virtual void setToConstant(double c) = 0;
  virtual void copyFrom(const hiopMatrixSparse& dm) = 0;

  virtual void copyRowsFrom(const hiopMatrix& src, const long long* rows_idxs, long long n_rows) = 0;

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;
  virtual void timesVec(double beta, double* y, double alpha, const double* x) const = 0;

  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;
  virtual void transTimesVec(double beta, double* y, double alpha, const double* x) const = 0;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_) = 0;
  virtual void addDiagonal(const double& value) = 0;
  virtual void addSubDiagonal(const double& alpha, long long start, const hiopVector& d_) = 0;
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, const hiopVector& d_,
    int start_on_src_vec, int num_elems = -1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X) = 0;

  /* block of W += alpha*this, where W is dense */
  virtual void addToSymDenseMatrixUpperTriangle(
    int row_dest_start, int col_dest_start, double alpha, hiopMatrixDense& W) const = 0;
  /* block of W += alpha*transpose(this), where W is dense */
  virtual void transAddToSymDenseMatrixUpperTriangle(
    int row_dest_start, int col_dest_start, double alpha, hiopMatrixDense& W) const = 0;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(
    int diag_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "counterpart method of hiopMatrixSymSparse should be used");
  }

  /* block of W += alpha*this, where W is sparse */
  virtual void addToSymSparseMatrixUpperTriangle(
    int row_dest_start, int col_dest_start, double alpha, hiopMatrixSparse& W) const = 0;
  /* block of W += alpha*transpose(this), where W is sparse */
  virtual void transAddToSymSparseMatrixUpperTriangle(
    int row_dest_start, int col_dest_start, double alpha, hiopMatrixSparse& W) const = 0;
  virtual void addUpperTriangleToSymSparseMatrixUpperTriangle(
    int diag_start, double alpha, hiopMatrixSparse& W) const
  {
    assert(false && "counterpart method of hiopMatrixSymSparse should be used");
  }

  /* diag block of W += alpha * M * D^{-1} * transpose(M), where M=this
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(
    int rowCol_dest_start, const double& alpha, const hiopVector& D, hiopMatrixDense& W) const = 0;

  /* block of W += alpha * M * D^{-1} * transpose(N), where M=this
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the
   * case, the method will assert(false) in debug; in release, the method will issue a
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(int row_dest_start, int col_dest_start,
    const double& alpha, const hiopVector& D, const hiopMatrixSparse& N, hiopMatrixDense& W) const = 0;

  virtual double max_abs_value() = 0;

  virtual bool isfinite() const = 0;

  // virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1,
    int rank = -1) const = 0;

  virtual hiopMatrix* alloc_clone() const = 0;
  virtual hiopMatrix* new_copy() const = 0;

  inline long long m() const
  {
    return nrows_;
  }
  inline long long n() const
  {
    return ncols_;
  }
  inline long long numberOfNonzeros() const
  {
    return nnz_;
  }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const
  {
    return false;
  }
  virtual bool checkIndexesAreOrdered() const = 0;
#endif
protected:
  int nrows_;   ///< number of rows
  int ncols_;   ///< number of columns
  int nnz_;     ///< number of nonzero entries
};

}   // namespace hiop
