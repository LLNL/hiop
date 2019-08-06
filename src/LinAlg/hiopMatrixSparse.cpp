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

/** res = alpha * this' * this 
* The method can work with @res being either an empty sparse matrix,
  i.e. hiopMatrixSparse(0,0,0.), in which case the storage is allocated
  and the sparse structure is created. In case @res already contains
  all the required storage space, we only update the numerical values
  of the nonzeros (assuming that the structure was set up previously)
*/
void hiopMatrixSparse::transTimesThis(double alpha, hiopMatrixSparse &res) const
{
  //create column respresentation of the matrix
  //TODO can be reused for all the subsequent calls
  vector<vector<int>> vvRows(ncols); // list of nnz row indices in each column
  vector<vector<double>> vvValues(ncols); // list of nnz values in each column

  for (int i = 0; i < nonzeroes; i++)
  {
    vvRows[jCol[i]].push_back(iRow[i]);
    vvValues[jCol[i]].push_back(values[i]);
  }

  // does the resulting sparse matrix needs to be initialized
  // or can we fill directly @res.values with the new values?
  bool resultNotInitialized = (res.nnz() == 0 && res.m()==0 && res.n()==0);
  
  //storage of the result in case @res is not initialized
  vector<vector<int>> vvRes_jCol(0);
  vector<vector<double>> vvRes_values(0);
  if (resultNotInitialized)
  {
    vvRes_jCol.resize(ncols);
    vvRes_values.resize(ncols);
  }
  // otherwise we can update directly @res
  int *res_iRow = res.get_iRow();
  int *res_jCol = res.get_jCol();
  double *res_values = res.get_values();
  int nnz_idx = 0;

  // compute dot product between columns c1 and c2
  for (int c1 = 0; c1 < ncols; c1++)
  {
    for (int c2 = 0; c2 < ncols; c2++) //c2=c1..ncols
    {
      auto rowIdx1 = vvRows[c1].begin();
      auto rowIdx2 = vvRows[c2].begin();
      auto value1  = vvValues[c1].begin();
      auto value2  = vvValues[c2].begin();
      double dot = 0.;
      bool newNonzero = false;

      while ( rowIdx1 != vvRows[c1].end() && rowIdx2 != vvRows[c2].end())
      {
        if (*rowIdx1 == *rowIdx2) //nonzeros at the same row index in both columns
        {
          dot += alpha * (*value1) * (*value2);
          rowIdx1++; rowIdx2++;
          value1++; value2++; 
          newNonzero = true;   
        } else if (*rowIdx1 < *rowIdx2)
        {
          rowIdx1++; value1++;
        } else 
        {
          rowIdx2++; value2++;
        } 
      }

      // process the new nonzero element
      if (newNonzero)
      {
        //we need to use auxiliary storage
        //the actual sparse matrix is created later 
        if (resultNotInitialized)
        {
          vvRes_jCol[c1].push_back(c2);
          vvRes_values[c1].push_back(dot);
        }
        //we can update directly @res.values
        else 
        {
          assert(nnz_idx < res.nnz());
          assert(res_iRow[nnz_idx] == c1);
          assert(res_jCol[nnz_idx] == c2);
          res_values[nnz_idx] = dot;
          nnz_idx++;
        }
      } 
    }
  }

  //construt the sparse matrix with the result if not done so previously
  if (resultNotInitialized)
    res.make(ncols, ncols, vvRes_jCol, vvRes_values);
}

void hiopMatrixSparse::make(int nrows_, int ncols_, vector<vector<int>> &vvCols, vector<vector<double>> &vvValues)
{
  assert(nonzeroes == 0);
  assert(nrows == 0);
  assert(ncols == 0);

  nrows = nrows_;
  ncols = ncols_;

  //count the number of nonzeros
  nonzeroes = 0;
  for (int i = 0; i < nrows_; i++)
  {
      nonzeroes += vvCols[i].size();
  }

  //allocate the space
  iRow   = new int[nonzeroes]; 
  jCol   = new int[nonzeroes]; 
  values = new double[nonzeroes]; 
  
  //fill in the structure and values
  int nnz_idx = 0;
  for (int i = 0; i < nrows_; i++)
  {
    auto itCols = vvCols[i].begin();
    auto itValues = vvValues[i].begin();
 
    while (itCols != vvCols[i].end())
    {
      iRow[nnz_idx] = i;
      jCol[nnz_idx] = *itCols;
      values[nnz_idx] = *itValues;
      nnz_idx++;
      itCols++;
      itValues++;
    }
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
