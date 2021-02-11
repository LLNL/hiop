#include "hiopMatrixSparseTriplet.hpp"
#include "hiopVectorPar.hpp"

#include "hiop_blasdefs.hpp"

#include <algorithm> //for std::min
#include <cmath> //for std::isfinite
#include <cstring>
#include <vector>
#include <numeric>
#include <cassert>

#include "hiopCppStdUtils.hpp"
namespace hiop
{

hiopMatrixSparseTriplet::hiopMatrixSparseTriplet(int rows, int cols, int nnz)
  : hiopMatrixSparse(rows, cols, nnz)
  , row_starts_(NULL)
{
  if(rows==0 || cols==0) {
    assert(nnz_==0 && "number of nonzeros must be zero when any of the dimensions are 0");
    nnz_ = 0;
  }

  iRow_ = new  int[nnz_];
  jCol_ = new int[nnz_];
  values_ = new double[nnz_];
}

hiopMatrixSparseTriplet::~hiopMatrixSparseTriplet()
{
  delete [] iRow_;
  delete [] jCol_;
  delete [] values_;
  delete row_starts_;
}

void hiopMatrixSparseTriplet::setToZero()
{
  for(int i=0; i<nnz_; i++)
    values_[i] = 0.;
}
void hiopMatrixSparseTriplet::setToConstant(double c)
{
  for(int i=0; i<nnz_; i++)
    values_[i] = c;
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseTriplet::timesVec(double beta,  hiopVector& y,
                                       double alpha, const hiopVector& x ) const
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
void hiopMatrixSparseTriplet::timesVec(double beta,  double* y,
                                       double alpha, const double* x ) const
{
  // y= beta*y
  for (int i = 0; i < nrows_; i++) {
    y[i] *= beta;
  }

  // y += alpha*this*x
  for (int i = 0; i < nnz_; i++) {
    assert(iRow_[i] < nrows_);
    assert(jCol_[i] < ncols_);
    y[iRow_[i]] += alpha * x[jCol_[i]] * values_[i];
  }
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseTriplet::transTimesVec(double beta,   hiopVector& y,
                                            double alpha,  const hiopVector& x ) const
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
void hiopMatrixSparseTriplet::transTimesVec(double beta,   double* y,
                                            double alpha,  const double* x ) const
{
  // y:= beta*y
  for (int i = 0; i < ncols_; i++) {
    y[i] *= beta;
  }

  // y += alpha*this^T*x
  for (int i = 0; i < nnz_; i++) {
    assert(iRow_[i] < nrows_);
    assert(jCol_[i] < ncols_);
    y[jCol_[i]] += alpha * x[iRow_[i]] * values_[i];
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

void hiopMatrixSparseTriplet::timesMatTrans(double beta, hiopMatrix& Wmat,
                                            double alpha, const hiopMatrix& M2mat) const
{
  auto& W = dynamic_cast<hiopMatrixDense&>(Wmat);
  const auto& M2 = dynamic_cast<const hiopMatrixSparseTriplet&>(M2mat);
  const hiopMatrixSparseTriplet& M1 = *this;
  const int m1 = M1.nrows_, nx = M1.ncols_, m2 = M2.nrows_;
  assert(nx==M1.ncols_);
  assert(nx==M2.ncols_);
  assert(M2.ncols_ == nx);

  assert(m1==W.m());
  assert(m2==W.n());
  
  double* WM = W.local_data();
  auto n_W = W.n();
  
  // TODO: allocAndBuildRowStarts -> should create row_starts internally (name='prepareRowStarts' ?)
  if(M1.row_starts_==NULL) M1.row_starts_ = M1.allocAndBuildRowStarts();
  assert(M1.row_starts_);

  if(M2.row_starts_==NULL) M2.row_starts_ = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_);

  double acc;

  for(int i=0; i<m1; i++) 
  {
    // j>=i
    for(int j=0; j<m2; j++)
    {
      acc = 0.;
      int ki=M1.row_starts_->idx_start_[i];
      int kj=M2.row_starts_->idx_start_[j];

      while(ki<M1.row_starts_->idx_start_[i+1] && kj<M2.row_starts_->idx_start_[j+1])
      {
        assert(ki<M1.nnz_);
        assert(kj<M2.nnz_);

        if(M1.jCol_[ki] == M2.jCol_[kj])
        {
          // same col, so multiply and increment 
          acc += M1.values_[ki] * M2.values_[kj];
          ki++;
          kj++;
        } 
        else if(M1.jCol_[ki]<M2.jCol_[kj])
        {
          // skip M1
          ki++;
        }
        else
        {
          // skip M2
          kj++;
        }                     
      } //end of loop over ki and kj
      WM[(i)*n_W + j] = beta*WM[(i)*n_W + j] + alpha*acc;
    } //end j
  } // end i    
}
void hiopMatrixSparseTriplet::addDiagonal(const double& alpha, const hiopVector& d_)
{
  assert(false && "not needed");
}
void hiopMatrixSparseTriplet::addDiagonal(const double& value)
{
  assert(false && "not needed");
}
void hiopMatrixSparseTriplet::addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseTriplet::copySubDiagonalFrom(const long long& start_on_dest_diag, const long long& num_elems,
                                                     const hiopVector& d_, const long long& start_on_nnz_idx, double scal)
{
  const hiopVectorPar& vd = dynamic_cast<const hiopVectorPar&>(d_);
  assert(num_elems<=vd.get_size());
  assert(start_on_dest_diag>=0 && start_on_dest_diag+num_elems<=this->nrows_);
  const double* v = vd.local_data_const();

  for(auto row_src=0; row_src<num_elems; row_src++) {
    const long long row_dest = row_src + start_on_dest_diag;
    const long long nnz_dest = row_src + start_on_nnz_idx;
    iRow_[nnz_dest] = jCol_[nnz_dest] = row_dest;
    this->values_[nnz_dest] = scal*v[row_src];
  }
}

void hiopMatrixSparseTriplet::setSubDiagonalTo(const long long& start_on_dest_diag, const long long& num_elems,
                                                      const double& c, const long long& start_on_nnz_idx)
{
  assert(start_on_dest_diag>=0 && start_on_dest_diag+num_elems<=this->nrows_);

  for(auto row_src=0; row_src<num_elems; row_src++) {
    const long long  row_dest = row_src + start_on_dest_diag;
    const long long  nnz_dest = row_src + start_on_nnz_idx;
    iRow_[nnz_dest] = row_dest;
    jCol_[nnz_dest] = row_dest;
    this->values_[nnz_dest] = c;
  }
}

void hiopMatrixSparseTriplet::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/* block of W += alpha*transpose(this)
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseTriplet::
transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start,
                                      double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols_<=W.m());
  assert(col_start>=0 && col_start+nrows_<=W.n());
  assert(W.n()==W.m());

  int m_W = W.m();
  double* WM = W.local_data();
  for(int it=0; it<nnz_; it++) {
    const int i = jCol_[it]+row_start;
    const int j = iRow_[it]+col_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "source entries need to map inside the upper triangular part of destination");
    //WM[i][j] += alpha*values_[it];
    WM[i*m_W+j] += alpha*values_[it];
  }
}

double hiopMatrixSparseTriplet::max_abs_value()
{
  char norm='M'; int one=1;
  double maxv = DLANGE(&norm, &one, &nnz_, values_, &one, NULL);
  return maxv;
}

bool hiopMatrixSparseTriplet::isfinite() const
{

#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  for(int i=0; i<nnz_; i++)
    if(false==std::isfinite(values_[i])) return false;
  return true;
}

hiopMatrixSparse* hiopMatrixSparseTriplet::alloc_clone() const
{
  return new hiopMatrixSparseTriplet(nrows_, ncols_, nnz_);
}

hiopMatrixSparse* hiopMatrixSparseTriplet::new_copy() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  hiopMatrixSparseTriplet* copy = new hiopMatrixSparseTriplet(nrows_, ncols_, nnz_);
  memcpy(copy->iRow_, iRow_, nnz_*sizeof(int));
  memcpy(copy->jCol_, jCol_, nnz_*sizeof(int));
  memcpy(copy->values_, values_, nnz_*sizeof(double));
  return copy;
}
void hiopMatrixSparseTriplet::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "this is to be implemented - method def too vague for now");
}

#ifdef HIOP_DEEPCHECKS
bool hiopMatrixSparseTriplet::checkIndexesAreOrdered() const
{
  if(nnz_==0) return true;
  for(int i=1; i<nnz_; i++) {
    if(iRow_[i] < iRow_[i-1]) return false;
    /* else */
    if(iRow_[i] == iRow_[i-1])
      if(jCol_[i] < jCol_[i-1]) return false;
  }
  return true;
}
#endif

void hiopMatrixSparseTriplet::
addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowAndCol_dest_start,
                                        const double& alpha,
                                        const hiopVector& D, hiopMatrixDense& W) const
{
  const int row_dest_start = rowAndCol_dest_start, col_dest_start = rowAndCol_dest_start;
  int n = this->nrows_;
  assert(row_dest_start>=0 && row_dest_start+n<=W.m());
  assert(col_dest_start>=0 && col_dest_start+nrows_<=W.n());
  assert(D.get_size() == this->ncols_);
  double* WM = W.local_data();
  int m_W = W.m();
  const double* DM = D.local_data_const();

  if(row_starts_==NULL) row_starts_ = allocAndBuildRowStarts();
  assert(row_starts_);

  double acc;

  for(int i=0; i<this->nrows_; i++) {
    //j==i
    acc = 0.;
    for(int k=row_starts_->idx_start_[i]; k<row_starts_->idx_start_[i+1]; k++)
      acc += this->values_[k] / DM[this->jCol_[k]] * this->values_[k];
    //WM[i+row_dest_start][i+col_dest_start] += alpha*acc;
    WM[(i+row_dest_start)*m_W+i+col_dest_start] += alpha*acc;

    //j>i
    for(int j=i+1; j<this->nrows_; j++) {
      //dest[i,j] = weigthed_dotprod(this_row_i,this_row_j)
      acc = 0.;

      int ki=row_starts_->idx_start_[i], kj=row_starts_->idx_start_[j];
      while(ki<row_starts_->idx_start_[i+1] && kj<row_starts_->idx_start_[j+1]) {
        assert(ki<this->nnz_);
        assert(kj<this->nnz_);
        if(this->jCol_[ki] == this->jCol_[kj]) {
          acc += this->values_[ki] / DM[this->jCol_[ki]] * this->values_[kj];
          ki++;
          kj++;
        } else {
          if(this->jCol_[ki]<this->jCol_[kj]) ki++;
          else                              kj++;
        }
      } //end of loop over ki and kj
      
      //WM[i+row_dest_start][j+col_dest_start] += alpha*acc;
      WM[(i+row_dest_start)*m_W + j+col_dest_start] += alpha*acc;
    } //end j
  } // end i

}

/*
 * block of W += alpha * M1 * D^{-1} * transpose(M2), where M1=this
 *  Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx)
 */
void hiopMatrixSparseTriplet::
addMDinvNtransToSymDeMatUTri(int row_dest_start, int col_dest_start,
                             const double& alpha,
                             const hiopVector& D, const hiopMatrixSparse& M2mat,
                             hiopMatrixDense& W) const
{
  const auto& M2 = dynamic_cast<const hiopMatrixSparseTriplet&>(M2mat);
  const hiopMatrixSparseTriplet& M1 = *this;
  const int m1 = M1.nrows_, nx = M1.ncols_, m2 = M2.nrows_;
  assert(nx==M1.ncols_);
  assert(nx==M2.ncols_);
  assert(D.get_size() == nx);
  assert(M2.ncols_ == nx);

  //does it fit in W ?
  assert(row_dest_start>=0 && row_dest_start+m1<=W.m());
  assert(col_dest_start>=0 && col_dest_start+m2<=W.n());

  double* WM = W.local_data();
  auto m_W = W.m();
  
  const double* DM = D.local_data_const();

  // TODO: allocAndBuildRowStarts -> should create row_starts internally (name='prepareRowStarts' ?)
  if(M1.row_starts_==NULL) M1.row_starts_ = M1.allocAndBuildRowStarts();
  assert(M1.row_starts_);

  if(M2.row_starts_==NULL) M2.row_starts_ = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_);

  double acc;

  // only parallelize these two outter loops
  //
  // sort amount of work per thread/exe unit
  // assign in order of most-> least work to better
  // distribute workload
  //
  // These are multiplied many times, but sparsity pattern
  // remains the same. We can do some preprocessing to save on
  // thread execution time
  //
  // compressed row/col patterns?
  for(int i=0; i<m1; i++) {
    // j>=i
    for(int j=0; j<m2; j++) {

      // dest[i,j] = weigthed_dotprod(M1_row_i,M2_row_j)
      acc = 0.;
      int ki=M1.row_starts_->idx_start_[i];
      int kj=M2.row_starts_->idx_start_[j];

      while(ki<M1.row_starts_->idx_start_[i+1] && kj<M2.row_starts_->idx_start_[j+1]) {
        assert(ki<M1.nnz_);
        assert(kj<M2.nnz_);

        if(M1.jCol_[ki] == M2.jCol_[kj]) {

          acc += M1.values_[ki] / DM[this->jCol_[ki]] * M2.values_[kj];
          ki++;
          kj++;
        } else {
          if(M1.jCol_[ki]<M2.jCol_[kj]) ki++;
          else                      kj++;
        }
      } //end of loop over ki and kj

#ifdef HIOP_DEEPCHECKS
      if(i+row_dest_start > j+col_dest_start)
        printf("[warning] lower triangular element updated in addMDinvNtransToSymDeMatUTri\n");
#endif
      assert(i+row_dest_start <= j+col_dest_start);
      //WM[i+row_dest_start][j+col_dest_start] += alpha*acc;
      WM[(i+row_dest_start)*m_W + j+col_dest_start] += alpha*acc;

    } //end j
  } // end i

}


// //assumes triplets are ordered
hiopMatrixSparseTriplet::RowStartsInfo*
hiopMatrixSparseTriplet::allocAndBuildRowStarts() const
{
  assert(nrows_>=0);

  RowStartsInfo* rsi = new RowStartsInfo(nrows_); assert(rsi);

  if(nrows_<=0) return rsi;

  int it_triplet=0;
  rsi->idx_start_[0]=0;
  for(int i=1; i<=this->nrows_; i++) {

    rsi->idx_start_[i]=rsi->idx_start_[i-1];

    while(it_triplet<this->nnz_ && this->iRow_[it_triplet]==i-1) {
#ifdef HIOP_DEEPCHECKS
      if(it_triplet>=1) {
        assert(iRow_[it_triplet-1]<=iRow_[it_triplet] && "row indexes are not sorted");
        //assert(iCol[it_triplet-1]<=iCol[it_triplet]);
        if(iRow_[it_triplet-1]==iRow_[it_triplet])
          assert(jCol_[it_triplet-1] < jCol_[it_triplet] && "col indexes are not sorted");
      }
#endif
      rsi->idx_start_[i]++;
      it_triplet++;
    }
    assert(rsi->idx_start_[i] == it_triplet);
  }
  assert(it_triplet==this->nnz_);
  return rsi;
}

void hiopMatrixSparseTriplet::copyRowsFrom(const hiopMatrix& src_gen,
                                           const long long* rows_idxs,
                                           long long n_rows)
{
  const hiopMatrixSparseTriplet& src = dynamic_cast<const hiopMatrixSparseTriplet&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  const int* iRow_src = src.i_row();
  const int* jCol_src = src.j_col();
  const double* values_src = src.M();
  int nnz_src = src.numberOfNonzeros();
  int itnz_src=0;
  int itnz_dest=0;
  //int iterators should suffice
  for(int row_dest=0; row_dest<n_rows; ++row_dest) {
    const int& row_src = rows_idxs[row_dest];

    while(itnz_src<nnz_src && iRow_src[itnz_src]<row_src) {
#ifdef HIOP_DEEPCHECKS
      if(itnz_src>0) {
        assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
        if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
          assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
#endif
      ++itnz_src;
    }

    while(itnz_src<nnz_src && iRow_src[itnz_src]==row_src) {
      assert(itnz_dest<nnz_);
#ifdef HIOP_DEEPCHECKS
      if(itnz_src>0) {
        assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
        if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
          assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
#endif
      iRow_[itnz_dest] = row_dest;//iRow_src[itnz_src];
      jCol_[itnz_dest] = jCol_src[itnz_src];
      values_[itnz_dest++] = values_src[itnz_src++];

      assert(itnz_dest<=nnz_);
    }
  }
  assert(itnz_dest == nnz_);
}

/**
 * @brief Copy 'n_rows' rows started from 'rows_src_idx_st' (array of size 'n_rows') from 'src' to the destination,
 * which starts from the 'rows_dest_idx_st'th row in 'this'
 *
 * @pre 'this' must have exactly, or more than 'n_rows' rows
 * @pre 'this' must have exactly, or more cols than 'src'
 */
void hiopMatrixSparseTriplet::copyRowsBlockFrom(const hiopMatrix& src_gen,
                                         const long long& rows_src_idx_st, const long long& n_rows,
                                         const long long& rows_dest_idx_st, const long long& dest_nnz_st)
{
  const hiopMatrixSparseTriplet& src = dynamic_cast<const hiopMatrixSparseTriplet&>(src_gen);
  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(this->n() >= src.n());
  assert(n_rows + rows_src_idx_st <= src.m());
  assert(n_rows + rows_dest_idx_st <= this->m());

  const int* iRow_src = src.i_row();
  const int* jCol_src = src.j_col();
  const double* values_src = src.M();
  int nnz_src = src.numberOfNonzeros();
  int itnz_src=0;
  int itnz_dest=dest_nnz_st;
  //int iterators should suffice
  for(auto row_add=0; row_add<n_rows; ++row_add) {
    const int row_src  = rows_src_idx_st  + row_add;
    const int row_dest = rows_dest_idx_st + row_add;

    while(itnz_src<nnz_src && iRow_src[itnz_src]<row_src) {
#ifdef HIOP_DEEPCHECKS
      if(itnz_src>0) {
      assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
      if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
        assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
#endif
      ++itnz_src;
    }

    while(itnz_src<nnz_src && iRow_src[itnz_src]==row_src) {
      assert(itnz_dest<nnz_);
#ifdef HIOP_DEEPCHECKS
      if(itnz_src>0) {
      assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
      if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
        assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
#endif
      iRow_[itnz_dest] = row_dest;//iRow_src[itnz_src];
      jCol_[itnz_dest] = jCol_src[itnz_src];
      values_[itnz_dest++] = values_src[itnz_src++];

      assert(itnz_dest<=nnz_);
    }
  }
}

void hiopMatrixSparseTriplet::
copyDiagMatrixToSubblock(const double& src_val,
                         const long long& dest_row_st, const long long& col_dest_st,
                         const long long& dest_nnz_st, const int &nnz_to_copy)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + col_dest_st <= this->n());

  int itnz_src=0;
  int itnz_dest=dest_nnz_st;
  for(auto ele_add=0; ele_add<nnz_to_copy; ++ele_add) {
    iRow_[itnz_dest] = dest_row_st + ele_add;
    jCol_[itnz_dest] = col_dest_st + ele_add;
    values_[itnz_dest++] = src_val;
  }
}

void hiopMatrixSparseTriplet::
copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                   const long long& dest_row_st, const long long& dest_col_st,
                                   const long long& dest_nnz_st, const int &nnz_to_copy,
                                   const hiopVector& ix)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());

  const hiopVectorPar& selected = dynamic_cast<const hiopVectorPar&>(ix);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(dx);
  const double *x=xx.local_data_const(), *pattern=selected.local_data_const();

  int dest_k = dest_nnz_st;
  int n = ix.get_local_size();
  int nnz_find=0;

  for(int i=0; i<n; i++)
  {
    if(pattern[i]!=0.0){
      iRow_[dest_k] = dest_row_st + nnz_find;
      jCol_[dest_k] = dest_col_st + nnz_find;
      values_[dest_k] = x[i];
      dest_k++;
      nnz_find++;
    }
  }
  assert(nnz_to_copy==nnz_find);
}

void hiopMatrixSparseTriplet::print(FILE* file, const char* msg/*=NULL*/,
                                    int maxRows/*=-1*/, int maxCols/*=-1*/,
                                    int rank/*=-1*/) const
{
  int myrank_=0, numranks=1; //this is a local object => always print

  if(file==NULL) file = stdout;

  int max_elems = maxRows>=0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);

  if(myrank_==rank || rank==-1) {

    if(NULL==msg) {
      if(numranks>1) {
        fprintf(file,
                "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n",
                m(), n(), numberOfNonzeros(), max_elems, myrank_);
      } else {
        fprintf(file,
                "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n",
                m(), n(), numberOfNonzeros(), max_elems);
      }
    } else {
      fprintf(file, "%s ", msg);
    }

    // using matlab indices
    fprintf(file, "iRow_=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", iRow_[it]+1);
    fprintf(file, "];\n");

    fprintf(file, "jCol_=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", jCol_[it]+1);
    fprintf(file, "];\n");

    fprintf(file, "v=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%22.16e; ", values_[it]);
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
  assert(ncols_ == nrows_);
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

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
  assert(ncols_ == nrows_);
  // y:= beta*y
  for (int i = 0; i < nrows_; i++) {
    y[i] *= beta;
  }

  // y += alpha*this*x
  for (int i = 0; i < nnz_; i++) {
    assert(iRow_[i] < nrows_);
    assert(jCol_[i] < ncols_);
    y[iRow_[i]] += alpha * x[jCol_[i]] * values_[i];
    if(iRow_[i]!=jCol_[i])
      y[jCol_[i]] += alpha * x[iRow_[i]] * values_[i];
  }
}

hiopMatrixSparse* hiopMatrixSymSparseTriplet::alloc_clone() const
{
  assert(nrows_ == ncols_);
  return new hiopMatrixSymSparseTriplet(nrows_, nnz_);
}
hiopMatrixSparse* hiopMatrixSymSparseTriplet::new_copy() const
{
  assert(nrows_ == ncols_);
  hiopMatrixSymSparseTriplet* copy = new hiopMatrixSymSparseTriplet(nrows_, nnz_);
  memcpy(copy->iRow_, iRow_, nnz_*sizeof(int));
  memcpy(copy->jCol_, jCol_, nnz_*sizeof(int));
  memcpy(copy->values_, values_, nnz_*sizeof(double));
  return copy;
}

/**
 * @brief block of W += alpha*this
 * @note W contains only the upper triangular entries
 */
void hiopMatrixSymSparseTriplet::
addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start,
                                              double alpha, hiopMatrixDense& W) const
{
  assert(diag_start>=0 && diag_start+nrows_<=W.m());
  assert(diag_start+ncols_<=W.n());
  assert(W.n()==W.m());

  const auto m_W = W.m();
  double* WM = W.local_data();
  for(int it=0; it<nnz_; it++) {
    assert(iRow_[it]<=jCol_[it] && "sparse symmetric matrices should contain only upper triangular entries");
    const int i = iRow_[it]+diag_start;
    const int j = jCol_[it]+diag_start;
    assert(i<W.m() && j<W.n()); assert(i>=0 && j>=0);
    assert(i<=j && "symMatrices not aligned; source entries need to map inside the upper triangular part of destination");
    //WM[i][j] += alpha*values_[it];
    WM[i*m_W+j] += alpha*values_[it];
  }
}

/**
 * @brief block of W += alpha*(this)^T
 * @note W contains only the upper triangular entries
 *
 * @warning This method should not be called directly.
 * Use addUpperTriangleToSymDenseMatrixUpperTriangle instead.
 */
void hiopMatrixSymSparseTriplet::
transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start,
                                      double alpha, hiopMatrixDense& W) const
{
  assert(0 && "This method should not be called for symmetric matrices.");
}

/* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
 * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
 * are available in 'vec_dest' starting at 'vec_start'
 */
void hiopMatrixSymSparseTriplet::startingAtAddSubDiagonalToStartingAt(int diag_src_start, const double& alpha,
                          hiopVector& vec_dest, int vec_start, int num_elems/*=-1*/) const
{
  hiopVectorPar& vd = dynamic_cast<hiopVectorPar&>(vec_dest);
  if(num_elems<0) num_elems = vd.get_size();
  assert(num_elems<=vd.get_size());

  assert(diag_src_start>=0 && diag_src_start+num_elems<=this->nrows_);
  double* v = vd.local_data();

  for(int itnz=0; itnz<nnz_; itnz++) {
    const int row = iRow_[itnz];
    if(row==jCol_[itnz]) {
      if(row>=diag_src_start && row<diag_src_start+num_elems) {
        assert(row+vec_start<vd.get_size());
        v[vec_start+row] += alpha * this->values_[itnz];
      }
    }
  }
}

void hiopMatrixSparseTriplet::copySubmatrixFrom(const hiopMatrix& src_gen,
                                   const long long& dest_row_st, const long long& dest_col_st,
                                   const long long& dest_nnz_st)
//                                   bool firstCall, std::map<int,int> &ValIdxMap )
{
  const hiopMatrixSparseTriplet& src = dynamic_cast<const hiopMatrixSparseTriplet&>(src_gen);
  auto m_rows = src.m();
  auto n_cols = src.n();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n() );
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  const int* src_iRow = src.i_row();
  const int* src_jCol = src.j_col();
  const double* src_val = src.M();
  int src_nnz = src.numberOfNonzeros();
  int dest_k = dest_nnz_st;

  // FIXME: irow and jcol only need to be assigned once; should we save a map for the indexes?
  for(auto src_k = 0; src_k < src_nnz; ++src_k,++dest_k) {
    iRow_[dest_k] = dest_row_st + src_iRow[src_k];
    jCol_[dest_k] = dest_col_st + src_jCol[src_k];
    values_[dest_k] = src_val[src_k];
  }
  assert(dest_k <= this->numberOfNonzeros());
}

void hiopMatrixSparseTriplet::copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                   const long long& dest_row_st, const long long& dest_col_st,
                                   const long long& dest_nnz_st)
{
  const hiopMatrixSparseTriplet& src = dynamic_cast<const hiopMatrixSparseTriplet&>(src_gen);
  auto m_rows = src.n();
  auto n_cols = src.m();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n() );
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  const int* src_iRow = src.j_col();
  const int* src_jCol = src.i_row();
  const double* src_val = src.M();
  int src_nnz = src.numberOfNonzeros();
  int dest_k = dest_nnz_st;

  // FIXME: irow and jcol only need to be assigned once; should we save a map for the indexes?
  for(auto src_k = 0; src_k < src_nnz; ++src_k,++dest_k) {
    iRow_[dest_k] = dest_row_st + src_iRow[src_k];
    jCol_[dest_k] = dest_col_st + src_jCol[src_k];
    values_[dest_k] = src_val[src_k];
  }
  assert(dest_k <= this->numberOfNonzeros());
}

void hiopMatrixSparseTriplet::setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                   const long long& dest_row_st, const long long& dest_col_st,
                                   const long long& dest_nnz_st, const int &nnz_to_copy, const hiopVector& ix)
{
  assert(ix.get_local_size() + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n() );
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  const hiopVectorPar& selected= dynamic_cast<const hiopVectorPar&>(ix);
  const double *pattern=selected.local_data_const();

  int dest_k = dest_nnz_st;
  int n = ix.get_local_size();
  int nnz_find=0;

  for(int i=0; i<n; i++){
    if(pattern[i]!=0.0){
      iRow_[dest_k] = dest_row_st + i;
      jCol_[dest_k] = dest_col_st + nnz_find;
      values_[dest_k] = scalar;
      nnz_find++;
      dest_k++;
    }
  }
  assert(nnz_find == nnz_to_copy);
}

void hiopMatrixSparseTriplet::setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                   const long long& dest_row_st, const long long& dest_col_st,
                                   const long long& dest_nnz_st, const int &nnz_to_copy, const hiopVector& ix)
{
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(ix.get_local_size() + dest_col_st <= this->n() );
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  const hiopVectorPar& selected= dynamic_cast<const hiopVectorPar&>(ix);
  const double *pattern = selected.local_data_const();

  int dest_k = dest_nnz_st;
  int n = ix.get_local_size();
  int nnz_find=0;

  for(int i=0; i<n; i++){
    if(pattern[i]!=0.0){
      iRow_[dest_k] = dest_row_st + nnz_find;
      jCol_[dest_k] = dest_col_st + i;
      values_[dest_k] = scalar;
      nnz_find++;
      dest_k++;
    }
  }
  assert(nnz_find == nnz_to_copy);
}


long long hiopMatrixSymSparseTriplet::numberOfOffDiagNonzeros()
{
  if(-1==nnz_offdiag_){
    nnz_offdiag_= nnz_;
    for(auto k=0;k<nnz_;k++){
      if(iRow_[k]==jCol_[k])
        nnz_offdiag_--;
    }
  }
  return nnz_offdiag_;
}


// Generate the three vectors A, IA, JA
void hiopMatrixSparseTriplet::convertToCSR(int &csr_nnz, int **csr_kRowPtr_in, int **csr_jCol_in, double **csr_kVal_in,
                                           int **index_covert_CSR2Triplet_in, int **index_covert_extra_Diag2CSR_in,
                                           std::unordered_map<int,int> &extra_diag_nnz_map)
{
  assert(*csr_kRowPtr_in==nullptr && *index_covert_CSR2Triplet_in==nullptr);
  int m = this->m();
  int n = this->n();
  int nnz = numberOfNonzeros();

  *csr_kRowPtr_in = new int[n+1]{};

  int *csr_kRowPtr = *csr_kRowPtr_in;

  csr_nnz = 0;
  /* transfer triplet form to CSR form
  * note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional diagonal elememts
  */
  int n_diag_val=0;
  std::unordered_map<int,int> extra_diag_nnz_map_temp;
  int *diag_defined = new int[n];

  // compute nnz in each row
  {
    for(int i=0;i<n;i++) diag_defined[i]=-1;
    // off-diagonal part
    csr_kRowPtr[0]=0;
    for(int k=0;k<nnz;k++){
      if(iRow_[k]!=jCol_[k]){
        csr_kRowPtr[iRow_[k]+1]++;
        csr_nnz++;
      }else{
        if(-1==diag_defined[iRow_[k]]){
          diag_defined[iRow_[k]] = k;
          csr_kRowPtr[iRow_[k]+1]++;
          csr_nnz++;
          n_diag_val++;
        }else{
          extra_diag_nnz_map_temp[iRow_[k]] = k;
        }
      }
    }
    // get correct row ptr index
    for(int i=1;i<n+1;i++){
      csr_kRowPtr[i] += csr_kRowPtr[i-1];
    }
    assert(csr_nnz==csr_kRowPtr[n]);
    assert(csr_nnz+extra_diag_nnz_map_temp.size()==nnz);

    *csr_kVal_in = new double[csr_nnz];
    *csr_jCol_in = new int[csr_nnz];
  }
  double *csr_kVal = *csr_kVal_in;
  int *csr_jCol = *csr_jCol_in;

  int *index_covert_extra_Diag2CSR_temp = new int[n];
  int *nnz_each_row_tmp = new int[n]{};

  // set correct col index and value
  {
    *index_covert_CSR2Triplet_in = new int[csr_nnz];
    *index_covert_extra_Diag2CSR_in = new int[n];

    int *index_covert_CSR2Triplet = *index_covert_CSR2Triplet_in;
    int *index_covert_extra_Diag2CSR = *index_covert_extra_Diag2CSR_in;

    for(int i=0;i<n;i++) diag_defined[i]=-1;

    int total_nnz_tmp{0},nnz_tmp{0}, rowID_tmp, colID_tmp;
    for(int k=0;k<n;k++){
        index_covert_extra_Diag2CSR_temp[k]=-1;
        index_covert_extra_Diag2CSR[k]=-1;
    }

    for(int k=0;k<nnz;k++){
      rowID_tmp = iRow_[k];
      colID_tmp = jCol_[k];
      if(rowID_tmp==colID_tmp){
        if(-1==diag_defined[rowID_tmp]){
          diag_defined[rowID_tmp] = k;
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + csr_kRowPtr[rowID_tmp];
          csr_jCol[nnz_tmp] = colID_tmp;
          csr_kVal[nnz_tmp] = values_[k];
          auto p = extra_diag_nnz_map_temp.find (rowID_tmp);
          if( p != extra_diag_nnz_map_temp.end() ){
            csr_kVal[nnz_tmp] += values_[p->second];
            index_covert_extra_Diag2CSR_temp[p->first] = nnz_tmp;
          }
          index_covert_CSR2Triplet[nnz_tmp] = k;

          nnz_each_row_tmp[rowID_tmp]++;
          total_nnz_tmp++;
        }
      }else{
        nnz_tmp = nnz_each_row_tmp[rowID_tmp] + csr_kRowPtr[rowID_tmp];
        csr_jCol[nnz_tmp] = colID_tmp;
        csr_kVal[nnz_tmp] = values_[k];
        index_covert_CSR2Triplet[nnz_tmp] = k;

        nnz_each_row_tmp[rowID_tmp]++;
        total_nnz_tmp++;
      }
    }

    // correct the missing diagonal term and sort the nonzeros
    for(int i=0;i<n;i++){
      // sort the nonzeros
      {
        std::vector<int> ind_temp(csr_kRowPtr[i+1]-csr_kRowPtr[i]);
        std::iota(ind_temp.begin(), ind_temp.end(), 0);
        std::sort(ind_temp.begin(), ind_temp.end(),[&](int a, int b){ return csr_jCol[a+csr_kRowPtr[i]]<csr_jCol[b+csr_kRowPtr[i]]; });

        reorder(csr_kVal+csr_kRowPtr[i],ind_temp,csr_kRowPtr[i+1]-csr_kRowPtr[i]);
        reorder(index_covert_CSR2Triplet+csr_kRowPtr[i],ind_temp,csr_kRowPtr[i+1]-csr_kRowPtr[i]);
        std::sort(csr_jCol+csr_kRowPtr[i],csr_jCol+csr_kRowPtr[i+1]);


        int old_nnz_idx = index_covert_extra_Diag2CSR_temp[i];
        if(old_nnz_idx!=-1){
          int old_nnz_in_row = ind_temp[old_nnz_idx - csr_kRowPtr[i]];
          std::vector<int>::iterator p = std::find(ind_temp.begin(),ind_temp.end(),old_nnz_in_row);
          assert(p != ind_temp.end());
          int new_nnz_idx = std::distance (ind_temp.begin(), p) + csr_kRowPtr[i];
          index_covert_extra_Diag2CSR[i] = new_nnz_idx;
          extra_diag_nnz_map[new_nnz_idx] = extra_diag_nnz_map_temp[i];
        }
      }
    }

  }

  delete [] nnz_each_row_tmp; nnz_each_row_tmp = nullptr;
  delete [] diag_defined; diag_defined = nullptr;
  delete [] index_covert_extra_Diag2CSR_temp; index_covert_extra_Diag2CSR_temp = nullptr;

}


} //end of namespace

