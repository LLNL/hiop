#include "hiopMatrixSparseTriplet.hpp"

#include "blasdefs.hpp"

#include <algorithm> //for std::min
#include <cmath> //for std::isfinite
#include <cstring>

#include <cassert>

namespace hiop
{

hiopMatrixSparseTriplet::hiopMatrixSparseTriplet(int rows, int cols, int nnz_)
  : nrows(rows), ncols(cols), nnz(nnz_)
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

void hiopMatrixSparseTriplet::setToZero()
{
  for(int i=0; i<nnz; i++)
    values[i] = 0.;
}
void hiopMatrixSparseTriplet::setToConstant(double c)
{
  for(int i=0; i<nnz; i++)
    values[i] = c;
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
  for (int i = 0; i < nnz; i++) {
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
  for (int i = 0; i < nnz; i++) {
    assert(iRow[i] < nrows);
    assert(jCol[i] < ncols);
    y[jCol[i]] += alpha * x[iRow[i]] * values[i];
  }
}

void hiopMatrixSparseTriplet::timesMat(double beta, hiopMatrix& W, 
				       double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseTriplet::transTimesMat(double beta, hiopMatrix& W, 
					    double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseTriplet::timesMatTrans(double beta, hiopMatrix& W, 
					    double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}
void hiopMatrixSparseTriplet::addDiagonal(const hiopVector& d_)
{
  assert(false && "not needed");
}
void hiopMatrixSparseTriplet::addDiagonal(const double& value)
{
  assert(false && "not needed");
}
void hiopMatrixSparseTriplet::addSubDiagonal(long long start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseTriplet::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/* block of W += alpha*this 
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseTriplet::addToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
							       double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+nrows<W.m());
  assert(col_start>=0 && col_start+ncols<W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int it=0; it<nnz; it++) {
    const int i = iRow[it]+row_start;
    const int j = jCol[it]+col_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "source entries need to map inside the upper triangular part of destination");
    WM[i][j] += alpha*values[it];
  }
}
/* block of W += alpha*transpose(this) 
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseTriplet::transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
								    double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols<W.m());
  assert(col_start>=0 && col_start+nrows<W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int it=0; it<nnz; it++) {
    const int i = jCol[it]+row_start;
    const int j = iRow[it]+col_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "source entries need to map inside the upper triangular part of destination");
    WM[i][j] += alpha*values[it];
  }
}

double hiopMatrixSparseTriplet::max_abs_value()
{
  char norm='M'; int one=1;
  double maxv = DLANGE(&norm, &one, &nnz, values, &one, NULL);
  return maxv;
}

bool hiopMatrixSparseTriplet::isfinite() const
{

#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  for(int i=0; i<nnz; i++)
    if(false==std::isfinite(values[i])) return false;
  return true;
}

hiopMatrix* hiopMatrixSparseTriplet::alloc_clone() const
{
  return new hiopMatrixSparseTriplet(nrows, ncols, nnz);
}

hiopMatrix* hiopMatrixSparseTriplet::new_copy() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  hiopMatrixSparseTriplet* copy = new hiopMatrixSparseTriplet(nrows, ncols, nnz);
  memcpy(copy->iRow, iRow, nnz*sizeof(int));
  memcpy(copy->jCol, jCol, nnz*sizeof(int));
  memcpy(copy->values, values, nnz*sizeof(double));
  return copy;
}
void hiopMatrixSparseTriplet::copyFrom(const hiopMatrixSparseTriplet& dm)
{
  assert(false && "this is to be implemented - method def too vague for now");
}

#ifdef HIOP_DEEPCHECKS
bool hiopMatrixSparseTriplet::checkIndexesAreOrdered() const
{
  if(nnz==0) return true;
  for(int i=1; i<nnz; i++) {
    if(iRow[i] < iRow[i-1]) return false;
    /* else */
    if(iRow[i] == iRow[i-1])
      if(jCol[i] < jCol[i-1]) return false;
  }
  return true;
}
#endif

// void hiopMatrixSparse::make(int nrows_, int ncols_, const vector<vector<int>> &vvCols, const vector<vector<double>> &vvValues)
// {
//   assert(nnz == 0);
//   assert(nrows == 0);
//   assert(ncols == 0);

//   nrows = nrows_;
//   ncols = ncols_;

//   //count the number of nonzeros
//   nnz = 0;
//   for (int i = 0; i < nrows_; i++)
//   {
//       nnz += vvCols[i].size();
//   }

//   //allocate the space
//   iRow   = new int[nnz]; 
//   jCol   = new int[nnz]; 
//   values = new double[nnz]; 
  
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

    int max_elems = maxRows>=0 ? maxRows : nnz;
    max_elems = std::min(max_elems, nnz);

  if(myrank==rank || rank==-1) {

    if(NULL==msg) {
      if(numranks>1)
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n", 
		nrows, ncols, nnz, max_elems, myrank);
      else
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n", 
		nrows, ncols, nnz, max_elems);
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

/**********************************************************************************
  * Sparse symmetric matrix in triplet format. Only the lower triangle is stored
  *********************************************************************************
*/
void hiopMatrixSymSparseTriplet::timesVec(double beta,  hiopVector& y,
					  double alpha, const hiopVector& x ) const
{
  assert(ncols == nrows);
  assert(x.get_size() == ncols);
  assert(y.get_size() == nrows);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}
 
/** y = beta * y + alpha * this * x */
void hiopMatrixSymSparseTriplet::timesVec(double beta,  double* y,
					  double alpha, const double* x ) const
{
  assert(ncols == nrows);
  // y:= beta*y
  for (int i = 0; i < nrows; i++) {
    y[i] *= beta;
  }

  // y += alpha*this*x
  for (int i = 0; i < nnz; i++) {
    assert(iRow[i] < nrows);
    assert(jCol[i] < ncols);
    y[iRow[i]] += alpha * x[jCol[i]] * values[i];
    if(iRow[i]!=jCol[i])
      y[jCol[i]] += alpha * x[iRow[i]] * values[i];
  }
}

hiopMatrix* hiopMatrixSymSparseTriplet::alloc_clone() const
{
  assert(nrows == ncols);
  return new hiopMatrixSymSparseTriplet(nrows, nnz);
}
hiopMatrix* hiopMatrixSymSparseTriplet::new_copy() const
{
  assert(nrows == ncols);
  hiopMatrixSymSparseTriplet* copy = new hiopMatrixSymSparseTriplet(nrows, nnz);
  memcpy(copy->iRow, iRow, nnz*sizeof(int));
  memcpy(copy->jCol, jCol, nnz*sizeof(int));
  memcpy(copy->values, values, nnz*sizeof(double));
  return copy;
}

/* block of W += alpha*this 
 * Note W; contains only the upper triangular entries */
void hiopMatrixSymSparseTriplet::addToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
						  double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+nrows<W.m());
  assert(col_start>=0 && col_start+ncols<W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int it=0; it<nnz; it++) {
    assert(iRow[it]<=jCol[it] && "sparse symmetric matrices should contain only upper triangular entries");
    const int i = iRow[it]+row_start;
    const int j = jCol[it]+col_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "symMatrices not aligned; source entries need to map inside the upper triangular part of destination");
    WM[i][j] += alpha*values[it];
  }
}
void hiopMatrixSymSparseTriplet::transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
								       double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols<W.m());
  assert(col_start>=0 && col_start+nrows<W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int it=0; it<nnz; it++) {
    assert(iRow[it]<=jCol[it] && "sparse symmetric matrices should contain only upper triangular entries");
    const int i = jCol[it]+row_start;
    const int j = iRow[it]+col_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "symMatrices not aligned; source entries need to map inside the upper triangular part of destination");
    WM[i][j] += alpha*values[it];
  }
}


} //end of namespace
