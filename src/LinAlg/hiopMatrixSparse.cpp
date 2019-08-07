#include "hiopMatrixSparse.hpp"

#include <cassert>
#include <iostream>

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

/** this = alpha * A' * A + beta*B
* The method can work with #this being either an empty sparse matrix,
  i.e. hiopMatrixSparse(0,0,0.), in which case the storage is allocated
  and the sparse structure is created. In case #this already contains
  all the required storage space, we only update the numerical values
  of the nonzeros (assuming that the structure was set up previously).
  
  \param[in] A is general nonsquare, nonsymmetric matrix
  \param[in] B is square symmetric matrix, containing only lower triangular part
  \param[in] alpha, beta are constants

  The method computes and returns only the lower triangular part of the symmetric result.
*/
void hiopMatrixSparse::transAAplusB(double alpha, const hiopMatrixSparse &A, double beta, const hiopMatrixSparse &B)
{
  assert(B.m() == B.n());
  assert(A.n() == B.m());
  
  // test if the structure of this matrix is initialized?
  // or can we fill directly #values with the new values?
  bool structureNotInitialized = (this->nnz() == 0 && this->m()==0 && this->n()==0);

  int nrows_A      = A.m();
  int ncols_A      = A.n();
  int nonzeroes_A  = A.nnz();
  const int *iRow_A      = A.get_iRow_const();
  const int *jCol_A      = A.get_jCol_const();
  const double *values_A = A.get_values_const();
  
  //create column respresentation of the matrix A
  //TODO can be reused for all the subsequent calls, except values_A
  vector<vector<int>> vvRows_A(ncols_A); // list of nnz row indices in each column
  vector<vector<double>> vvValues_A(ncols_A); // list of nnz values in each column
  for (int i = 0; i < nonzeroes_A; i++)
  {
    vvRows_A[jCol_A[i]].push_back(iRow_A[i]);
    vvValues_A[jCol_A[i]].push_back(values_A[i]);
  }
  
  //storage of the result in case #this is not initialized
  vector<vector<int>> vvCols_Result(0);
  vector<vector<double>> vvValues_Result(0);
  if (structureNotInitialized)
  {
    vvCols_Result.resize(ncols_A);
    vvValues_Result.resize(ncols_A);
  }
  // otherwise we can update directly #this.values
  int nnz_idx = 0;

  // properties and iterator in matrix B
  const int *iRow_B      = B.get_iRow_const();
  const int *jCol_B      = B.get_jCol_const();
  const double *values_B = B.get_values_const();
  int nonzeroes_B        = B.nnz();
  int nnz_idx_B = 0;

  // compute alpha*A'A + beta*B
  for (int c1 = 0; c1 < ncols_A; c1++)
  {
    for (int c2 = c1; c2 < ncols_A; c2++) //compute only lower triangular part
    {
      //TODO: skip empty 
      //if (vvRows_A[c1].begin() == vvRows_A.end() && zero @B[c1,c2])

      auto rowIdx1 = vvRows_A[c1].begin();
      auto rowIdx2 = vvRows_A[c2].begin();
      auto value1  = vvValues_A[c1].begin();
      auto value2  = vvValues_A[c2].begin();
      double dot = 0.;
      bool newNonzero = false;

      //compute alpha * A' * A
      while ( rowIdx1 != vvRows_A[c1].end() && rowIdx2 != vvRows_A[c2].end())
      {
        if (*rowIdx1 == *rowIdx2) //nonzeros at the same row index in both columns
        {
          // compute dot product between columns c1 and c2
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

      // add nonzeros from beta*B, B is lower triangular NLP hessian
      if (nnz_idx_B < nonzeroes_B &&
          iRow_B[nnz_idx_B] == c1 &&
          jCol_B[nnz_idx_B] == c2)
      {
        dot += beta * values_B[nnz_idx_B];
        newNonzero = true;
        nnz_idx_B++;
      }

      // process the new nonzero element
      if (newNonzero)
      {
        //we need to use auxiliary storage
        //the actual sparse matrix is created later 
        if (structureNotInitialized)
        {
          vvCols_Result[c1].push_back(c2);
          vvValues_Result[c1].push_back(dot);
        }
        //we can update directly #values
        else 
        {
          assert(0); //did not test this path
          assert(nnz_idx < nonzeroes);
          assert(iRow[nnz_idx] == c1);
          assert(jCol[nnz_idx] == c2);
          values[nnz_idx] = dot;
          nnz_idx++;
        }
      } 
    }
  }

  //construt the sparse matrix with the result if not done so previously
  if (structureNotInitialized)
    this->make(ncols_A, ncols_A, vvCols_Result, vvValues_Result);
}

void hiopMatrixSparse::make(int nrows_, int ncols_, const vector<vector<int>> &vvCols, const vector<vector<double>> &vvValues)
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
