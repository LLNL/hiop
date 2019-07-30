#ifndef HIOP_MATRIX_SPARSE
#define HIOP_MATRIX_SPARSE

#include "hiop_defs.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 

#ifndef MPI_COMM
#define MPI_Comm int
#endif
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF 0
#endif
#include <cstddef>

#endif 

#include <cstdio>

namespace hiop
{

class hiopVector;
class hiopVectorPar;

/*! \class hiopMatrixSparse
    \brief A sparse matrix in triplet format.

    Sparse matrix represented by a triplet format
    #iRow, #jRow, #values all of which are arrays of
    size #nonzeroes.

    // this particular 2x4 Matrix is dense
    iRow[0] = 0; jCol[0] = 0; values[0] = 1.0; 
    iRow[1] = 0; jCol[1] = 1; values[1] = 1.0;
    iRow[2] = 0; jCol[2] = 2; values[2] = 1.0;
    iRow[3] = 0; jCol[3] = 3; values[3] = 1.0;
    iRow[4] = 1; jCol[4] = 0; values[4] = 1.0;
    iRow[5] = 1; jCol[5] = 1; values[5] = 1.0;
    iRow[6] = 1; jCol[6] = 2; values[6] = 1.0;
    iRow[7] = 1; jCol[7] = 3; values[7] = 1.0;
*/
//TODO: this class should inherit from the hiopMatrix
class hiopMatrixSparse
{
public:
  hiopMatrixSparse(int rows, int cols, int nnz);
  ~hiopMatrixSparse();
 

  //TODO: having hiopVector& interface instead of double* would
  //require to keep additional vector at hiopAugLagrNlpAdapter
  //class, specifically at functions eval_grad_f() and eval_grad_Lagr()
  //and copy the vectors data to ouput raw double* pointer.
//  /** y = beta * y + alpha * this * x */
//  void timesVec(double beta,  hiopVector& y,
//			double alpha, const hiopVector& x ) const;
//
//  /** y = beta * y + alpha * this^T * x */
//  void transTimesVec(double beta,   hiopVector& y,
//			     double alpha,  const hiopVector& x ) const;
  
  /** y = beta * y + alpha * this * x */
  void timesVec(double beta,  double* y,
			double alpha, const double* x ) const;

  /** y = beta * y + alpha * this^T * x */
  void transTimesVec(double beta,   double* y,
			     double alpha,  const double* x ) const;

  /* call with -1 to print all rows, all columns, or on all ranks; otherwise will
  *  will print the first rows and/or columns on the specified rank.
  */
  //void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;
  
  /* number of rows */
  int m() const {return nrows;}
  /* number of columns */
  int n() const {return ncols;}
  /* number of nonzeroes */
  int nnz() const {return nonzeroes;}


  /* pointer to the iRow */
  int* get_iRow() {return iRow;}
  /* pointer to the jCol */
  int* get_jCol() {return jCol;}
  /* pointer to the values */
  double* get_values() {return values;}

  void print(FILE* file, const char* msg=NULL, int max_elems=-1, int rank=-1) const; 


#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  //bool assertSymmetry(double tol=1e-16) const;
#endif

private:
  int nrows; ///< number of rows
  int ncols; ///< number of columns
  int nonzeroes;  ///< number of nonzero entries
   
  int* iRow; ///< row idices of the nonzero entries
  int* jCol; ///< column indices of the nonzero entries
  double* values; ///< values of the nonzero entries

};


}
#endif
