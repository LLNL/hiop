#include "hiopMatrixSparse.hpp"

#include <cassert>

namespace hiop
{

hiopMatrixSparse::hiopMatrixSparse(int rows, int cols, int nnz):
    nrows(rows),
    ncols(cols),
    nonzeroes(nnz),
    iRow(new int[nnz]),
    jCol(new int[nnz]),
    values(new double[nnz])
{}

hiopMatrixSparse::~hiopMatrixSparse()
{
    delete [] iRow;
    delete [] jCol;
    delete [] values;
}

//  /** y = beta * y + alpha * this * x */
//  void hiopMatrixSparse::timesVec(double beta,  hiopVector& y,
//   	                            double alpha, const hiopVector& x ) const
//  {
//      assert(x.get_size() == ncols);
//      assert(y.get_size() == nrows);
//  
//      double* y_data = y.local_data();
//      const double* x_data = x.local_data_const();
//  
//      // y:= beta*y
//      y.scale(beta);
//  
//      // y += alpha*this*x
//      for (int i = 0; i < nonzeroes; i++)
//      {
//          y_data[iRow[i]] += alpha * x_data[jCol[i]] * values[i];
//      }
//  }
//  
//  
//  /** y = beta * y + alpha * this^T * x */
//  void hiopMatrixSparse::transTimesVec(double beta,   hiopVector& y,
//                              	     double alpha,  const hiopVector& x ) const
//  {
//      assert(x.get_size() == nrows);
//      assert(y.get_size() == ncols);
//  
//      double* y_data = y.local_data();
//      const double* x_data = x.local_data_const();
//  
//      // y:= beta*y
//      y.scale(beta);
//  
//      // y += alpha*this^T*x
//      for (int i = 0; i < nonzeroes; i++)
//      {
//          y_data[jCol[i]] += alpha * x_data[iRow[i]] * values[i];
//      }
//  }

/** y = beta * y + alpha * this * x */
void hiopMatrixSparse::timesVec(double beta,  double* y,
 	                            double alpha, const double* x ) const
{
    // y:= beta*y
    for (int i = 0; i < nrows; i++) 
    {
        y[i] *= beta;
    }

    // y += alpha*this*x
    for (int i = 0; i < nonzeroes; i++)
    {
        assert(iRow[i] < nrows);
        assert(jCol[i] < ncols);
        y[iRow[i]] += alpha * x[jCol[i]] * values[i];
    }
}


/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparse::transTimesVec(double beta,   double* y,
                            	     double alpha,  const double* x ) const
{
    // y:= beta*y
    for (int i = 0; i < ncols; i++) 
    {
        y[i] *= beta;
    }

    // y += alpha*this^T*x
    for (int i = 0; i < nonzeroes; i++)
    {
        assert(iRow[i] < nrows);
        assert(jCol[i] < ncols);
        y[jCol[i]] += alpha * x[iRow[i]] * values[i];
    }
}
}
