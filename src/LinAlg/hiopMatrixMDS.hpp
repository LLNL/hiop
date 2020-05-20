#ifndef HIOP_SPARSE_MATRIX_MDS
#define HIOP_SPARSE_MATRIX_MDS

#include "hiopVector.hpp"
#include "hiopMatrix.hpp"
#include "hiopMatrixSparseTriplet.hpp"

#include <algorithm>

#include <cassert>

namespace hiop
{

/** Mixed Sparse-Dense blocks matrix  - it is not distributed
 *  M = [S D] where S is sparse and D is dense
*/
class hiopMatrixMDS : public hiopMatrix
{
public:
  hiopMatrixMDS(int rows, int cols_sparse, int cols_dense, int nnz_sparse)
  {
    mSp = new hiopMatrixSparseTriplet(rows, cols_sparse, nnz_sparse);
    mDe = new hiopMatrixDense(rows, cols_dense);
  }
  virtual ~hiopMatrixMDS()
  {
    delete mDe;
    delete mSp;
  }

  virtual void setToZero()
  {
    mSp->setToZero();
    mDe->setToZero();
  }
  virtual void setToConstant(double c)
  {
    mSp->setToConstant(c);
    mDe->setToConstant(c);
  }
  virtual void copyFrom(const hiopMatrixMDS& m) 
  {
    mSp->copyFrom(*m.mSp);
    mDe->copyFrom(*m.mDe);
  }

  virtual void copyRowsFrom(const hiopMatrix& src_in, const long long* rows_idxs, long long n_rows)
  {
    const hiopMatrixMDS& src = dynamic_cast<const hiopMatrixMDS&>(src_in);
    mSp->copyRowsFrom(*src.mSp, rows_idxs, n_rows);
    mDe->copyRowsFrom(*src.mDe, rows_idxs, n_rows);
  }
  
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const
  {
    hiopVectorPar* yp = dynamic_cast<hiopVectorPar*>(&y);
    const hiopVectorPar* xp = dynamic_cast<const hiopVectorPar*>(&x);
    assert(yp);
    assert(xp);
    assert(xp->get_size() == mSp->n()+mDe->n());
    mSp->timesVec(beta, yp->local_data(), alpha, xp->local_data_const());
    mDe->timesVec(1.,   yp->local_data(), alpha, xp->local_data_const()+mSp->n());
  }
  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const
  {
    hiopVectorPar* yp = dynamic_cast<hiopVectorPar*>(&y);
    const hiopVectorPar* xp = dynamic_cast<const hiopVectorPar*>(&x);
    assert(yp);
    assert(xp);
    assert(yp->get_size() == mSp->n()+mDe->n());
    mSp->transTimesVec(beta, yp->local_data(),          alpha, xp->local_data_const());
    mDe->transTimesVec(beta, yp->local_data()+mSp->n(), alpha, xp->local_data_const());
  }

  /* W = beta*W + alpha*this*X */  
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }


  virtual void addDiagonal(const double& alpha, const hiopVector& d_)
  {
    assert(false && "not supported");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not supported");
  }
  virtual void addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
  {
    assert(false && "not supported");
  }
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X)
  {
    const hiopMatrixMDS* pX=dynamic_cast<const hiopMatrixMDS*>(&X);
    if(pX==NULL) {
      assert(false && "operation only supported for hiopMatrixMDS left operand");
    }
    mSp->addMatrix(alpha, *pX->mSp);
    mDe->addMatrix(alpha, *pX->mDe);
  }

  /* block of W += alpha*this */
  virtual void addToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    mSp->addToSymDenseMatrixUpperTriangle(row_start, col_start, alpha, W);
    mDe->addToSymDenseMatrixUpperTriangle(row_start, col_start+mSp->n(), alpha, W);
  }
  /* block of W += alpha*this' */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    mSp->transAddToSymDenseMatrixUpperTriangle(row_start,          col_start, alpha, W);
    mDe->transAddToSymDenseMatrixUpperTriangle(row_start+mSp->n(), col_start, alpha, W);
  }

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   * 
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   * 
   * Preconditions: 
   *  1. this->n()==this-m()
   *  2. W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							     double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not for general matrices; counterpart method from hiopMatrixSymBlockDiagMDS should be used instead");
  }

  virtual double max_abs_value()
  {
    return std::max(mSp->max_abs_value(), mDe->max_abs_value());
  }

  virtual bool isfinite() const
  {
    return mSp->isfinite() && mDe->isfinite();
  }
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const
  {
    mSp->print(f,msg,maxRows,maxCols,rank);
    mDe->print(f,msg,maxRows,maxCols,rank);
  }

  virtual hiopMatrix* alloc_clone() const
  {
    hiopMatrixMDS* m = new hiopMatrixMDS();
    assert(m->mSp==NULL); assert(m->mDe==NULL); 
    m->mSp = dynamic_cast<hiopMatrixSparseTriplet*>(mSp->alloc_clone());
    m->mDe = dynamic_cast<hiopMatrixDense*>(mDe->alloc_clone());
    assert(m->mSp!=NULL); assert(m->mDe!=NULL); 
    return m;
  }
  virtual hiopMatrix* new_copy() const
  {
    hiopMatrixMDS* m = new hiopMatrixMDS();
    assert(m->mSp==NULL); assert(m->mDe==NULL); 
    m->mSp = dynamic_cast<hiopMatrixSparseTriplet*>(mSp->new_copy());
    m->mDe = dynamic_cast<hiopMatrixDense*>(mDe->new_copy());
    assert(m->mSp!=NULL); assert(m->mDe!=NULL); 
    return m;
  }

  virtual inline long long m() const {return mSp->m();}
  virtual inline long long n() const {return mSp->n()+mDe->n();}
  inline long long n_sp() const {return mSp->n();}
  inline long long n_de() const {return  mDe->n();}

  inline const hiopMatrixSparseTriplet* sp_mat() const { return mSp; }
  inline const hiopMatrixDense* de_mat() const { return mDe; }

  inline int sp_nnz() const { return mSp->numberOfNonzeros(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double** de_local_data() { return mDe->local_data(); }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return false; }
#endif
private:
  hiopMatrixSparseTriplet* mSp;
  hiopMatrixDense* mDe;
private:
  hiopMatrixMDS() : mSp(NULL), mDe(NULL) {};
  hiopMatrixMDS(const hiopMatrixMDS&) {};
};

class hiopMatrixSymBlockDiagMDS : public hiopMatrix
{
public:
  hiopMatrixSymBlockDiagMDS(int n_sparse, int n_dense, int nnz_sparse)
  {
    mSp = new hiopMatrixSymSparseTriplet(n_sparse, nnz_sparse);
    mDe = new hiopMatrixDense(n_dense, n_dense);
  }
  virtual ~hiopMatrixSymBlockDiagMDS()
  {
    delete mDe;
    delete mSp;
  }

  virtual void setToZero()
  {
    mSp->setToZero();
    mDe->setToZero();
  }
  virtual void setToConstant(double c)
  {
    mSp->setToConstant(c);
    mDe->setToConstant(c);
  }
  virtual void copyFrom(const hiopMatrixSymBlockDiagMDS& m) 
  {
    mSp->copyFrom(*m.mSp);
    mDe->copyFrom(*m.mDe);
  }

  virtual void copyRowsFrom(const hiopMatrix& src_in, const long long* rows_idxs, long long n_rows)
  {
    const hiopMatrixSymBlockDiagMDS& src = dynamic_cast<const hiopMatrixSymBlockDiagMDS&>(src_in);
    mSp->copyRowsFrom(src, rows_idxs, n_rows);
    mDe->copyRowsFrom(src, rows_idxs, n_rows);
  }
  
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const
  {
    hiopVectorPar* yp = dynamic_cast<hiopVectorPar*>(&y);
    const hiopVectorPar* xp = dynamic_cast<const hiopVectorPar*>(&x);
    assert(yp);
    assert(xp);
 
    assert(xp->get_size() == mSp->n()+mDe->n());
    assert(yp->get_size() == mSp->n()+mDe->n());

    mSp->timesVec(beta, yp->local_data(),          alpha, xp->local_data_const());
    mDe->timesVec(beta, yp->local_data()+mSp->n(), alpha, xp->local_data_const()+mSp->n());
  }
  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const
  {
    timesVec(beta, y, alpha, x);
  }

  /* W = beta*W + alpha*this*X */  
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }


  virtual void addDiagonal(const double& alpha, const hiopVector& d_)
  {
    assert(false && "not supported");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not supported");
  }
  virtual void addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
  {
    assert(false && "not supported");
  }

  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X)
  {
    const hiopMatrixSymBlockDiagMDS* pX=dynamic_cast<const hiopMatrixSymBlockDiagMDS*>(&X);
    if(pX==NULL) {
      assert(false && "operation only supported for hiopMatrixMDS left operand");
    } 
    mSp->addMatrix(alpha, *pX->mSp);
    mDe->addMatrix(alpha, *pX->mDe);
  }

  /* block of W += alpha*this */
  virtual void addToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(mSp->m() == mSp->n());
    mSp->addToSymDenseMatrixUpperTriangle(row_start,          col_start,          alpha, W);
    mDe->addToSymDenseMatrixUpperTriangle(row_start+mSp->n(), col_start+mSp->n(), alpha, W);
  }
  /* block of W += alpha*this */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(mSp->m() == mSp->n());
    mSp->transAddToSymDenseMatrixUpperTriangle(row_start,          col_start,          alpha, W);
    mDe->transAddToSymDenseMatrixUpperTriangle(row_start+mSp->n(), col_start+mSp->n(), alpha, W);
  }

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   * 
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   * 
   * Preconditions: 
   *  1. this->n()==this-m()
   *  2. W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							     double alpha, hiopMatrixDense& W) const
  {
    assert(mSp->m() == mSp->n());
    mSp->addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start,          alpha, W);
    mDe->addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start+mSp->m(), alpha, W);
  }

  virtual double max_abs_value()
  {
    return std::max(mSp->max_abs_value(), mDe->max_abs_value());
  }

  virtual bool isfinite() const
  {
    return mSp->isfinite() && mDe->isfinite();
  }
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const
  {
    mSp->print(f,msg,maxRows,maxCols,rank);
    mDe->print(f,msg,maxRows,maxCols,rank);
  }

  virtual hiopMatrix* alloc_clone() const
  {
    hiopMatrixSymBlockDiagMDS* m = new hiopMatrixSymBlockDiagMDS();
    assert(m->mSp==NULL); assert(m->mDe==NULL); 
    m->mSp = dynamic_cast<hiopMatrixSymSparseTriplet*>(mSp->alloc_clone());
    m->mDe = dynamic_cast<hiopMatrixDense*>(mDe->alloc_clone());
    assert(m->mSp!=NULL); assert(m->mDe!=NULL); 
    return m;
  }
  virtual hiopMatrix* new_copy() const
  {
    hiopMatrixSymBlockDiagMDS* m = new hiopMatrixSymBlockDiagMDS();
    assert(m->mSp==NULL); assert(m->mDe==NULL); 
    m->mSp = dynamic_cast<hiopMatrixSymSparseTriplet*>(mSp->new_copy());
    m->mDe = dynamic_cast<hiopMatrixDense*>(mDe->new_copy());
    assert(m->mSp!=NULL); assert(m->mDe!=NULL); 
    return m;
  }

  virtual inline long long m() const {return n();}
  virtual inline long long n() const {return mSp->n()+mDe->n();}
  inline long long n_sp() const {return mSp->n();}
  inline long long n_de() const {return  mDe->n();}

  inline const hiopMatrixSymSparseTriplet* sp_mat() const { return mSp; }
  inline const hiopMatrixDense* de_mat() const { return mDe; }

  inline int sp_nnz() const { return mSp->numberOfNonzeros(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double** de_local_data() { return mDe->local_data(); }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const
  {
    if(mSp->assertSymmetry(tol))
      return mDe->assertSymmetry(tol);
    else
      return false;
  }
#endif
private:
  hiopMatrixSymSparseTriplet* mSp;
  hiopMatrixDense* mDe;
private:
  hiopMatrixSymBlockDiagMDS() : mSp(NULL), mDe(NULL) {};
  hiopMatrixSymBlockDiagMDS(const hiopMatrixMDS&) {};
};

} //end of namespace

#endif
