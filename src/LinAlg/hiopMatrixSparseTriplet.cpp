#include "hiopMatrixSparseTriplet.hpp"

#include <algorithm> //for std::min
#include <cassert>

namespace hiop
{

hiopMatrixSparseTriplet::hiopMatrixSparseTriplet(int rows, int cols, int nnz)
  : nrows(rows), ncols(cols), nonzeroes(nnz)
{
  iRow = new  int[nnz];
  jCol = new int[nnz];
  values = new double[nnz];
}

hiopMatrixSparseTriplet::~hiopMatrixSparseTriplet()
{
  delete [] iRow;
  delete [] jCol;
  delete [] values;
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseTriplet::timesVec(double beta,  hiopVector& y,
				double alpha, const hiopVector& x ) const
{
  assert(x.get_size() == ncols);
  assert(y.get_size() == nrows);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}
 
/** y = beta * y + alpha * this * x */
void hiopMatrixSparseTriplet::timesVec(double beta,  double* y,
				       double alpha, const double* x ) const
{
    // y:= beta*y
  for (int i = 0; i < nrows; i++) {
    y[i] *= beta;
  }

  // y += alpha*this*x
  for (int i = 0; i < nonzeroes; i++) {
    assert(iRow[i] < nrows);
    assert(jCol[i] < ncols);
    y[iRow[i]] += alpha * x[jCol[i]] * values[i];
  }
}
 
/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseTriplet::transTimesVec(double beta,   hiopVector& y,
                             	     double alpha,  const hiopVector& x ) const
{
  assert(x.get_size() == nrows);
  assert(y.get_size() == ncols);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);
  
  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();
  
  transTimesVec(beta, y_data, alpha, x_data);
}
 
/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseTriplet::transTimesVec(double beta,   double* y,
					    double alpha,  const double* x ) const
{
  // y:= beta*y
  for (int i = 0; i < ncols; i++) {
    y[i] *= beta;
  }
  
  // y += alpha*this^T*x
  for (int i = 0; i < nonzeroes; i++) {
    assert(iRow[i] < nrows);
    assert(jCol[i] < ncols);
    y[jCol[i]] += alpha * x[iRow[i]] * values[i];
  }
}

// void hiopMatrixSparse::make(int nrows_, int ncols_, const vector<vector<int>> &vvCols, const vector<vector<double>> &vvValues)
// {
//   assert(nonzeroes == 0);
//   assert(nrows == 0);
//   assert(ncols == 0);

//   nrows = nrows_;
//   ncols = ncols_;

//   //count the number of nonzeros
//   nonzeroes = 0;
//   for (int i = 0; i < nrows_; i++)
//   {
//       nonzeroes += vvCols[i].size();
//   }

//   //allocate the space
//   iRow   = new int[nonzeroes]; 
//   jCol   = new int[nonzeroes]; 
//   values = new double[nonzeroes]; 
  
//   //fill in the structure and values
//   int nnz_idx = 0;
//   for (int i = 0; i < nrows_; i++)
//   {
//     auto itCols = vvCols[i].begin();
//     auto itValues = vvValues[i].begin();
 
//     while (itCols != vvCols[i].end())
//     {
//       iRow[nnz_idx] = i;
//       jCol[nnz_idx] = *itCols;
//       values[nnz_idx] = *itValues;
//       nnz_idx++;
//       itCols++;
//       itValues++;
//     }
//   }
// }

void hiopMatrixSparseTriplet::print(FILE* file, const char* msg/*=NULL*/, 
				    int maxRows/*=-1*/, int maxCols/*=-1*/, 
				    int rank/*=-1*/) const 
{
  int myrank=0, numranks=1; //this is a local object => always print

    int max_elems = maxRows>=0 ? maxRows : nonzeroes;
    max_elems = std::min(max_elems, nonzeroes);

  if(myrank==rank || rank==-1) {

    if(NULL==msg) {
      if(numranks>1)
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n", 
		nrows, ncols, nonzeroes, max_elems, myrank);
      else
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n", 
		nrows, ncols, nonzeroes, max_elems);
    } else {
      fprintf(file, "%s ", msg);
    }    



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
 } //end of namespace
