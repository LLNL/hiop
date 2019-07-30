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

void hiopMatrixSparse::print(FILE* file, const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const 
{
  int myrank=0, numranks=1; 
//#ifdef HIOP_USE_MPI
//  if(rank>=0) {
//    auto comm = MPI_COM_WORLD;
//    int err = MPI_Comm_rank(comm, &myrank); assert(err==MPI_SUCCESS);
//    err = MPI_Comm_size(comm, &numranks); assert(err==MPI_SUCCESS);
//  }
//#endif
  if(myrank==rank || rank==-1) {
    if(max_elems>nonzeroes) max_elems=nonzeroes;

    if(NULL==msg) {
      if(numranks>1)
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n", nrows, ncols, nonzeroes, max_elems, myrank);
      else
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n", nrows, ncols, nonzeroes, max_elems);
    } else {
      fprintf(file, "%s ", msg);
    }    

    max_elems = max_elems>=0?max_elems:nonzeroes;
    // using matlab indices
    fprintf(file, "iRow=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d ; ", iRow[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "jCol=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d ; ", jCol[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "v=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%22.16e ; ", values[it]);
    fprintf(file, "];\n");
  }
}


}
