#pragma once
#include "hiopMatrix.hpp"
#include <cstddef>
#include <cstdio>
#include <cassert>

namespace hiop
{
class hiopMatrixDenseRowMajor;
/** Dense matrix stored row-wise and distributed column-wise 
 */
class hiopMatrixDense : public hiopMatrix
{
public:
  hiopMatrixDense(const long long& m, const long long& glob_n, MPI_Comm _comm = MPI_COMM_SELF)
      : m_local(m)
      , n_global(glob_n)
      , comm(_comm)
  {
  }
  virtual ~hiopMatrixDense()
  {
  }

  virtual void setToZero(){assert(false && "not implemented in base class");}
  virtual void setToConstant(double c){assert(false && "not implemented in base class");}
  virtual void copyFrom(const hiopMatrixDense& dm){assert(false && "not implemented in base class");}
  virtual void copyFrom(const double* buffer){assert(false && "not implemented in base class");}

  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const{assert(false && "not implemented in base class");}
  /* same as above for mostly internal use - avoid using it */
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const{assert(false && "not implemented in base class");}

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const{assert(false && "not implemented in base class");}
  /* same as above for mostly for internal use - avoid using it */
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const{assert(false && "not implemented in base class");}
  /** W = beta*W + alpha*this*X 
   *
   * Precondition: W, 'this', and 'X' need to be local matrices (not distributed). All multiplications 
   * of distributed matrices needed by HiOp internally can be done efficiently in parallel using the 
   * 'timesMatTrans' and 'transTimesMat' methods below.
   */ 
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const{assert(false && "not implemented in base class");}
  
  /** W = beta*W + alpha*this*X 
   * Contains the implementation internals of the above; can be used on its own.
   */
  virtual void timesMat_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const{assert(false && "not implemented in base class");}

  /** W = beta*W + alpha*this^T*X 
   * Precondition: 'this' should be local/non-distributed. 'X' (and 'W') can be distributed.
   *
   * Note: no inter-process communication occurs in the parallel case
   */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const{assert(false && "not implemented in base class");}

  /** W = beta*W + alpha*this*X^T 
   * Precondition: 'W' need to be local/non-distributed.
   *
   * 'this' and 'X' can be distributed, in which case communication will occur.
   */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const{assert(false && "not implemented in base class");}
  /* Contains dgemm wrapper needed by the above */
  virtual void timesMatTrans_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const{assert(false && "not implemented in base class");}

  virtual void addDiagonal(const double& alpha, const hiopVector& d_){assert(false && "not implemented in base class");}
  virtual void addDiagonal(const double& value){assert(false && "not implemented in base class");}
  virtual void addSubDiagonal(const double& alpha, long long start_on_dest_diag, const hiopVector& d_){assert(false && "not implemented in base class");}
  
  /** add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1){assert(false && "not implemented in base class");}
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c){assert(false && "not implemented in base class");}
  
  virtual void addMatrix(double alpha, const hiopMatrix& X){assert(false && "not implemented in base class");}

  /* block of W += alpha*this
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions: 
   *  1. 'this' has to fit in the upper triangle of W 
   *  2. W.n() == W.m()
   */
  virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						double alpha, hiopMatrixDense& W) const{assert(false && "not implemented in base class");}
  /* block of W += alpha*transpose(this)
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions: 
   *  1. transpose of 'this' has to fit in the upper triangle of W 
   *  2. W.n() == W.m()
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						     double alpha, hiopMatrixDense& W) const{assert(false && "not implemented in base class");}

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   * 
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   * 
   * Preconditions: 
   *  1. this->n()==this->m()
   *  2. W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							     double alpha, hiopMatrixDense& W) const{assert(false && "not implemented in base class");}

  virtual double max_abs_value(){assert(false && "not implemented in base class");}

  virtual bool isfinite() const{assert(false && "not implemented in base class");}
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const{assert(false && "not implemented in base class");}

  virtual hiopMatrixDense* alloc_clone() const=0;//{assert(false && "not implemented in base class");}
  virtual hiopMatrixDense* new_copy() const=0;//{assert(false && "not implemented in base class");}

  virtual void appendRow(const hiopVector& row){assert(false && "not implemented in base class");}
  /*copies the first 'num_rows' rows from 'src' to 'this' starting at 'row_dest' */
  virtual void copyRowsFrom(const hiopMatrixDense& src, int num_rows, int row_dest){assert(false && "not implemented in base class");}
  
  /* Copy 'n_rows' rows specified by 'rows_idxs' (array of size 'n_rows') from 'src' to 'this'
   * 
   * Preconditions
   * 1. 'this' has exactly 'n_rows' rows
   * 2. 'src' and 'this' must have same number of columns
   * 3. number of rows in 'src' must be at least the number of rows in 'this'
   */
  virtual void copyRowsFrom(const hiopMatrix& src_gen, const long long* rows_idxs, long long n_rows){assert(false && "not implemented in base class");}
  
  /* copies 'src' into this as a block starting at (i_block_start,j_block_start) */
  virtual void copyBlockFromMatrix(const long i_block_start, const long j_block_start,
			   const hiopMatrixDense& src){assert(false && "not implemented in base class");}
  
  /* overwrites 'this' with 'src''s block that starts at (i_src_block_start,j_src_block_start) 
   * and has dimensions of 'this' */
  virtual void copyFromMatrixBlock(const hiopMatrixDense& src, const int i_src_block_start, const int j_src_block_start){assert(false && "not implemented in base class");}
  /*  shift<0 -> up; shift>0 -> down  */
  virtual void shiftRows(long long shift){assert(false && "not implemented in base class");}
  virtual void replaceRow(long long row, const hiopVector& vec){assert(false && "not implemented in base class");}
  /* copies row 'irow' in the vector 'row_vec' (sizes should match) */
  virtual void getRow(long long irow, hiopVector& row_vec){assert(false && "not implemented in base class");}
#ifdef HIOP_DEEPCHECKS
  virtual void overwriteUpperTriangleWithLower(){assert(false && "not implemented in base class");}
  virtual void overwriteLowerTriangleWithUpper(){assert(false && "not implemented in base class");}
#endif
  virtual long long get_local_size_n() const {assert(false && "not implemented in base class");}
  virtual long long get_local_size_m() const {assert(false && "not implemented in base class");}
  virtual MPI_Comm get_mpi_comm() const { return comm; }

  //TODO: this is not kosher!
  virtual double** local_data() const {assert(false && "not implemented in base class");}
  virtual double*  local_buffer() const {assert(false && "not implemented in base class");}
  //do not use this unless you sure you know what you're doing
  virtual double** get_M(){assert(false && "not implemented in base class");}

  virtual long long m() const {return m_local;}
  virtual long long n() const {return n_global;}
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const{assert(false && "not implemented in base class");}
#endif
protected:
  long long n_global; //total / global number of columns
  int m_local;
  MPI_Comm comm; 
  int myrank;

protected:
  hiopMatrixDense() {};
  /** copy constructor, for internal/private use only (it doesn't copy the values) */
  hiopMatrixDense(const hiopMatrixDense&){assert(false && "not implemented in base class");}
};

} // namespace hiop

