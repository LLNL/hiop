#include "hiopMatrixSparseCSR.hpp"
#include "hiopVectorPar.hpp"

#include "hiop_blasdefs.hpp"

#include <algorithm> //for std::min
#include <cmath> //for std::isfinite
#include <cstring>
#include <vector>
#include <numeric>
#include <cassert>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include "hiopCppStdUtils.hpp"

namespace hiop
{

hiopMatrixSparseCSR::hiopMatrixSparseCSR(size_type rows, size_type cols, size_type nnz)
  : hiopMatrixSparse(rows, cols, nnz),
    buf_col_(nullptr)
{
  if(rows==0 || cols==0) {
    assert(nnz_==0 && "number of nonzeros must be zero when any of the dimensions are 0");
    nnz_ = 0;
  }

  irowptr_ = new index_type[nrows_];
  jcolind_ = new index_type[nnz_];
  values_ = new double[nnz_];
}

hiopMatrixSparseCSR::~hiopMatrixSparseCSR()
{
  delete[] buf_col_;
  delete[] irowptr_;
  delete[] jcolind_;
  delete[] values_;
}

void hiopMatrixSparseCSR::setToZero()
{
  for(index_type i=0; i<nnz_; i++)
    values_[i] = 0.;
}
void hiopMatrixSparseCSR::setToConstant(double c)
{
  for(index_type i=0; i<nnz_; i++)
    values_[i] = c;
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSR::timesVec(double beta,
                                   hiopVector& y,
                                   double alpha,
                                   const hiopVector& x) const
{
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSR::timesVec(double beta,
                                   double* y,
                                   double alpha,
                                   const double* x) const
{
  assert(false && "not yet implemented");
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSR::transTimesVec(double beta,
                                            hiopVector& y,
                                            double alpha,
                                            const hiopVector& x) const
{
  assert(x.get_size() == nrows_);
  assert(y.get_size() == ncols_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  transTimesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSR::transTimesVec(double beta,
                                        double* y,
                                        double alpha,
                                        const double* x) const
{
  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSR::timesMat(double beta,
                                   hiopMatrix& W,
                                   double alpha,
                                   const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::transTimesMat(double beta,
                                            hiopMatrix& W,
                                            double alpha,
                                            const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::timesMatTrans(double beta,
                                        hiopMatrix& Wmat,
                                        double alpha,
                                        const hiopMatrix& M2mat) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::addDiagonal(const double& alpha, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::addDiagonal(const double& value)
{
  assert(false && "not needed");
}
void hiopMatrixSparseCSR::addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                              const size_type& num_elems,
                                              const hiopVector& d_,
                                              const index_type& start_on_nnz_idx,
                                              double scal)
{
  assert(false && "not implemented");
}

void hiopMatrixSparseCSR::setSubDiagonalTo(const index_type& start_on_dest_diag,
                                           const size_type& num_elems,
                                           const double& c,
                                           const index_type& start_on_nnz_idx)
{
  assert(start_on_dest_diag>=0 && start_on_dest_diag+num_elems<=this->nrows_);
  assert(false && "not implemented");
}

void hiopMatrixSparseCSR::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/* block of W += alpha*transpose(this)
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseCSR::
transAddToSymDenseMatrixUpperTriangle(index_type row_start,
                                      index_type col_start,
                                      double alpha,
                                      hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols_<=W.m());
  assert(col_start>=0 && col_start+nrows_<=W.n());
  assert(W.n()==W.m());

  assert(false && "not yet implemented");
}

double hiopMatrixSparseCSR::max_abs_value()
{
  char norm='M'; size_type one=1;
  double maxv = DLANGE(&norm, &one, &nnz_, values_, &one, nullptr);
  return maxv;
}

void hiopMatrixSparseCSR::row_max_abs_value(hiopVector &ret_vec)
{
  assert(ret_vec.get_local_size() == nrows_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(ret_vec);
  yy.setToZero();
  double* y_data = yy.local_data();

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSR::scale_row(hiopVector &vec_scal, const bool inv_scale)
{
  assert(vec_scal.get_local_size() == nrows_);

  hiopVectorPar& vscal = dynamic_cast<hiopVectorPar&>(vec_scal);  
  double* vd = vscal.local_data();
  assert(false && "not yet implemented");
}

bool hiopMatrixSparseCSR::isfinite() const
{
  for(index_type i=0; i<nnz_; i++)
    if(false==std::isfinite(values_[i])) return false;
  return true;
}

hiopMatrixSparse* hiopMatrixSparseCSR::alloc_clone() const
{
  return new hiopMatrixSparseCSR(nrows_, ncols_, nnz_);
}

hiopMatrixSparse* hiopMatrixSparseCSR::new_copy() const
{
  hiopMatrixSparseCSR* copy = new hiopMatrixSparseCSR(nrows_, ncols_, nnz_);
  memcpy(copy->irowptr_, irowptr_, (nrows_+1)*sizeof(index_type));
  memcpy(copy->jcolind_, jcolind_, nnz_*sizeof(index_type));
  memcpy(copy->values_, values_, nnz_*sizeof(double));
  return copy;
}
void hiopMatrixSparseCSR::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "to be implemented - method def too vague for now");
}

/// @brief copy to 3 arrays.
/// @pre these 3 arrays are not nullptr
void hiopMatrixSparseCSR::copy_to(index_type* irow, index_type* jcol, double* val)
{
  assert(irow && jcol && val);
  memcpy(irow, irowptr_, (1+nrows_)*sizeof(index_type));
  memcpy(jcol, jcolind_, nnz_*sizeof(index_type));
  memcpy(val, values_, nnz_*sizeof(double));
}

void hiopMatrixSparseCSR::copy_to(hiopMatrixDense& W)
{
  assert(false && "not needed");
  assert(W.m() == nrows_);
  assert(W.n() == ncols_);
}

void hiopMatrixSparseCSR::
addMDinvMtransToDiagBlockOfSymDeMatUTri(index_type rowAndCol_dest_start,
                                        const double& alpha,
                                        const hiopVector& D, hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

/*
 * block of W += alpha * M1 * D^{-1} * transpose(M2), where M1=this
 *  Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx)
 */
void hiopMatrixSparseCSR::
addMDinvNtransToSymDeMatUTri(index_type row_dest_start,
                             index_type col_dest_start,
                             const double& alpha,
                             const hiopVector& D,
                             const hiopMatrixSparse& M2mat,
                             hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSR::copyRowsFrom(const hiopMatrix& src_gen,
                                       const index_type* rows_idxs,
                                       size_type n_rows)
{
  const hiopMatrixSparseCSR& src = dynamic_cast<const hiopMatrixSparseCSR&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  assert(false && "not yet implemented");
}

/**
 * @brief Copy 'n_rows' rows started from 'rows_src_idx_st' (array of size 'n_rows') from 'src' to the destination,
 * which starts from the 'rows_dest_idx_st'th row in 'this'
 *
 * @pre 'this' must have exactly, or more than 'n_rows' rows
 * @pre 'this' must have exactly, or more cols than 'src'
 */
void hiopMatrixSparseCSR::copyRowsBlockFrom(const hiopMatrix& src_gen,
                                         const index_type& rows_src_idx_st, const size_type& n_rows,
                                         const index_type& rows_dest_idx_st, const size_type& dest_nnz_st)
{
  const hiopMatrixSparseCSR& src = dynamic_cast<const hiopMatrixSparseCSR&>(src_gen);
  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(this->n() >= src.n());
  assert(n_rows + rows_src_idx_st <= src.m());
  assert(n_rows + rows_dest_idx_st <= this->m());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSR::
copyDiagMatrixToSubblock(const double& src_val,
                         const index_type& dest_row_st,
                         const index_type& col_dest_st,
                         const size_type& dest_nnz_st,
                         const size_type &nnz_to_copy)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + col_dest_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSR::
copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                   const index_type& dest_row_st,
                                   const index_type& dest_col_st,
                                   const size_type& dest_nnz_st,
                                   const size_type &nnz_to_copy,
                                   const hiopVector& ix)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSR::print(FILE* file, const char* msg/*=nullptr*/,
                                    int maxRows/*=-1*/, int maxCols/*=-1*/,
                                    int rank/*=-1*/) const
{
  int myrank_=0, numranks=1; //this is a local object => always print

  if(file==nullptr) file = stdout;

  int max_elems = maxRows>=0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);
  
  if(myrank_==rank || rank==-1) {
    std::stringstream ss;
    if(nullptr==msg) {
      if(numranks>1) {
        ss << "CSR matrix of size " << m() << " " << n() << " and nonzeros " 
           << numberOfNonzeros() << ", printing " <<  max_elems << " elems (on rank="
           << myrank_ << ")" << std::endl;
      } else {
        ss << "CSR matrix of size " << m() << " " << n() << " and nonzeros " 
           << numberOfNonzeros() << ", printing " <<  max_elems << " elems" << std::endl;
      }
    } else {
      ss << msg << " ";
    }

    // using matlab indices (starting at 1)
    //fprintf(file, "iRow_=[");
    ss << "iRow_=[";

    for(index_type i=0; i<nrows_; i++) {
      const index_type ip1 = i+1;
      for(int p=irowptr_[i]; p<irowptr_[i+1] && p<max_elems; ++p) {
        ss << ip1 << "; ";
      }
    }
    ss << "];" << std::endl;

    ss << "jCol_=[";
    for(index_type it=0; it<max_elems; it++) {
      ss << (jcolind_[it]+1) << "; ";
    }
    ss << "];" << std::endl;
    
    ss << "v=[";
    ss << std::scientific << std::setprecision(16);
    for(index_type it=0; it<max_elems; it++) {
      ss << values_[it] << "; ";
    }
    //fprintf(file, "];\n");
    ss << "];" << std::endl;

    fprintf(file, "%s", ss.str().c_str());
  }
}


//M = X*D*Y -> computes nnz in M and allocates M 
//By convention, M is mxn, X is mxK and Y is Kxn
hiopMatrixSparseCSR* hiopMatrixSparseCSR::times_mat_alloc(const hiopMatrixSparseCSR& Y) const
{
  const index_type* irowptrY = Y.i_row();
  const index_type* jcolindY = Y.j_col();

  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;

  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);
  
  index_type nnzM = 0;
    // count the number of entries in the result M
  char* flag = new char[m];
  for(int i=0; i<m; i++) {
    //reset flag 
    memset(flag, 0, m*sizeof(char));

    for(int pt=irowptrX[i]; pt<irowptrX[i+1]; pt++) {
      //X[i,k] is nonzero
      const index_type k = jcolindX[pt];
      assert(k<K);

      //add the nonzero pattern of row k of Y to M
      for(int p=irowptrY[k]; p<irowptrY[k+1]; p++) {
	const index_type j = jcolindY[p];
        assert(j<n);
        
        //Y[k,j] is non zero, hence M[i,j] is non zero
	if(flag[j]==0) {
          //only count once
	  nnzM++;
	  flag[j]=1;
	}
      }
    }
  }
  assert(nnzM>=0); //overflow?!?

  delete[] flag;

  //allocate result M
  return new hiopMatrixSparseCSR(m, n, nnzM);
} 

/**
 *  M = X*D*Y -> computes nnz in M and allocates M 
 * By convention, M is mxn, X is mxK, Y is Kxn, and D is size K.
 * 
 * The algorithm uses the fact that the sparsity pattern of the i-th row of M is
 *           K
 * M_{i*} = sum x_{ik} Y_{j*}   (see Tim Davis book p.17)
 *          k=1
 * Therefore, to get sparsity pattern of the i-th row of M:
 *  1. we iterate over nonzeros (i,k) in the i-th row of X
 *  2. for each such k we iterate over the nonzeros (k,j) in the k-th row of Y and 
 *  3. count (i,j) as nonzero of M 
 */
void hiopMatrixSparseCSR::times_mat_symbolic(hiopMatrixSparseCSR& M,
                                             const hiopMatrixSparseCSR& Y) const
{
  const index_type* irowptrY = Y.i_row();
  const index_type* jcolindY = Y.j_col();
  const double* valuesY = Y.M();
  
  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;
  const double* valuesX = values_;

  index_type* irowptrM = M.i_row();
  index_type* jcolindM = M.j_col();
  double* valuesM = M.M();
  
  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);

  if(nullptr == M.buf_col_) {
    M.buf_col_ = new double[n];
  }
  double* W = M.buf_col_;
  
  char* flag=new char[n];

  for(int it=0; it<n; it++) {
    W[it] = 0.0;
  }

  int nnzM=0;
  for(int i=0; i<m; i++) {
    memset(flag, 0, m);

    assert(nnzM<M.numberOfNonzeros());
    //start row i of M
    irowptrM[i]=nnzM;
    
    for(int px=irowptrX[i]; px<irowptrX[i+1]; px++) { 
      const auto k = jcolindX[px]; //X[i,k] is non-zero
      assert(k<K);
      
      //const double val = valuesX[px]*d[k];

      //iterate the row k of Y and scatter the values into W
      for(int py=irowptrY[k]; py<irowptrY[k+1]; py++) {
	const auto j = jcolindY[py];
        assert(j<n);
        
	//we have M[k,j] nonzero; add it if not already added
	if(flag[j]==0) {
          assert(nnzM<M.numberOfNonzeros());
          
	  jcolindM[nnzM++]=j;
	  flag[j]=1;
	}
        //W[j] += (valuesY[py]*val);
      }
    }
  }
  irowptrM[n] = nnzM;
  delete[] flag;
}

void hiopMatrixSparseCSR::times_mat_numeric(double beta,
                                            hiopMatrixSparseCSR& M,
                                            double alpha,
                                            const hiopMatrixSparseCSR& Y)
{
  const index_type* irowptrY = Y.i_row();
  const index_type* jcolindY = Y.j_col();
  const double* valuesY = Y.M();
  
  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;
  const double* valuesX = values_;

  index_type* irowptrM = M.i_row();
  index_type* jcolindM = M.j_col();
  double* valuesM = M.M();
  
  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);

  if(nullptr == M.buf_col_) {
    M.buf_col_ = new double[n];
  }
  double* W = M.buf_col_;

  for(int it=0; it<n; it++) {
    W[it] = 0.0;
  }

  for(int i=0; i<m; i++) {
    for(int px=irowptrX[i]; px<irowptrX[i+1]; px++) { 
      const auto k = jcolindX[px]; //X[i,k] is non-zero
      assert(k<K);
      
      const double val = valuesX[px];

      //iterate the row k of Y and scatter the values into W
      for(int py=irowptrY[k]; py<irowptrY[k+1]; py++) {
        assert(jcolindY[py]<n);        
	W[jcolindY[py]] += (valuesY[py]*val);
      }
    }
    //gather the values into the i-th row M
    for(int p=irowptrM[i]; p<irowptrM[i+1]; ++p) {
      const auto j = jcolindM[p];
      valuesM[p] = W[j];
      W[j] = 0.0;
    }
  }
}

} //end of namespace

