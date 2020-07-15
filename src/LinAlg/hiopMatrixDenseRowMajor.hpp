#pragma once
#include "hiopMatrixDense.hpp"
#include <cstddef>
#include <cstdio>

namespace hiop
{
/** Dense matrix stored row-wise and distributed column-wise 
 */
class hiopMatrixDenseRowMajor : public hiopMatrixDense
{
public:

  hiopMatrixDenseRowMajor(const long long& m, 
		  const long long& glob_n, 
		  long long* col_part=NULL, 
		  MPI_Comm comm=MPI_COMM_SELF, 
		  const long long& m_max_alloc=-1);
  virtual ~hiopMatrixDenseRowMajor();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixDense& dm);
  virtual void copyFrom(const double* buffer);

  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  /* same as above for mostly internal use - avoid using it */
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const;
  /* same as above for mostly for internal use - avoid using it */
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const;
  /** W = beta*W + alpha*this*X 
   *
   * Precondition: W, 'this', and 'X' need to be local matrices (not distributed). All multiplications 
   * of distributed matrices needed by HiOp internally can be done efficiently in parallel using the 
   * 'timesMatTrans' and 'transTimesMat' methods below.
   */ 
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  
  /** W = beta*W + alpha*this*X 
   * Contains the implementation internals of the above; can be used on its own.
   */
  virtual void timesMat_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /** W = beta*W + alpha*this^T*X 
   * Precondition: 'this' should be local/non-distributed. 'X' (and 'W') can be distributed.
   *
   * Note: no inter-process communication occurs in the parallel case
   */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /** W = beta*W + alpha*this*X^T 
   * Precondition: 'W' need to be local/non-distributed.
   *
   * 'this' and 'X' can be distributed, in which case communication will occur.
   */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  /* Contains dgemm wrapper needed by the above */
  virtual void timesMatTrans_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, long long start_on_dest_diag, const hiopVector& d_);
  
  /** add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1);
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c);
  
  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /* block of W += alpha*this
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions: 
   *  1. 'this' has to fit in the upper triangle of W 
   *  2. W.n() == W.m()
   */
  virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						double alpha, hiopMatrixDense& W) const;
  /* block of W += alpha*transpose(this)
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions: 
   *  1. transpose of 'this' has to fit in the upper triangle of W 
   *  2. W.n() == W.m()
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						     double alpha, hiopMatrixDense& W) const;

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
							     double alpha, hiopMatrixDense& W) const;

  virtual double max_abs_value();

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual hiopMatrixDense* alloc_clone() const;
  virtual hiopMatrixDense* new_copy() const;

  void appendRow(const hiopVector& row);
  /*copies the first 'num_rows' rows from 'src' to 'this' starting at 'row_dest' */
  void copyRowsFrom(const hiopMatrixDense& src, int num_rows, int row_dest);
  
  /* Copy 'n_rows' rows specified by 'rows_idxs' (array of size 'n_rows') from 'src' to 'this'
   * 
   * Preconditions
   * 1. 'this' has exactly 'n_rows' rows
   * 2. 'src' and 'this' must have same number of columns
   * 3. number of rows in 'src' must be at least the number of rows in 'this'
   */
  void copyRowsFrom(const hiopMatrix& src_gen, const long long* rows_idxs, long long n_rows);
  
  /* copies 'src' into this as a block starting at (i_block_start,j_block_start) */
  void copyBlockFromMatrix(const long i_block_start, const long j_block_start,
			   const hiopMatrixDense& src);
  
  /* overwrites 'this' with 'src''s block that starts at (i_src_block_start,j_src_block_start) 
   * and has dimensions of 'this' */
  void copyFromMatrixBlock(const hiopMatrixDense& src, const int i_src_block_start, const int j_src_block_start);
  /*  shift<0 -> up; shift>0 -> down  */
  void shiftRows(long long shift);
  void replaceRow(long long row, const hiopVector& vec);
  /* copies row 'irow' in the vector 'row_vec' (sizes should match) */
  void getRow(long long irow, hiopVector& row_vec);
#ifdef HIOP_DEEPCHECKS
  void overwriteUpperTriangleWithLower();
  void overwriteLowerTriangleWithUpper();
#endif
  virtual long long get_local_size_n() const { return n_local; }
  virtual long long get_local_size_m() const { return m_local; }
  virtual MPI_Comm get_mpi_comm() const { return comm; }

  //TODO: this is not kosher!
  inline double** local_data() const {return M; }
  inline double*  local_buffer() const {return M[0]; }
  //do not use this unless you sure you know what you're doing
  inline double** get_M() { return M; }

  virtual long long m() const {return m_local;}
  virtual long long n() const {return n_global;}
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
#endif
private:
  double** M; //local storage
  int n_local; //local number of rows and cols, respectively
  long long glob_jl, glob_ju;

  mutable double* _buff_mxnlocal;  

  //this is very private do not touch :)
  long long max_rows;
private:
  hiopMatrixDenseRowMajor() {};
  /** copy constructor, for internal/private use only (it doesn't copy the values) */
  hiopMatrixDenseRowMajor(const hiopMatrixDenseRowMajor&);

  inline double* new_mxnlocal_buff() const {
    if(_buff_mxnlocal==NULL) {
      _buff_mxnlocal = new double[max_rows*n_local];
    } 
    return _buff_mxnlocal;
  }
};

} // namespace hiop

