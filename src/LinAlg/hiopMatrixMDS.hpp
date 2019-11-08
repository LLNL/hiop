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

  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const
  {
    hiopVectorPar* yp = dynamic_cast<hiopVectorPar*>(&y);
    const hiopVectorPar* xp = dynamic_cast<const hiopVectorPar*>(&x);
    assert(yp);
    assert(xp);
    assert(xp->get_size() == mSp->n()+mDe->n());
    mSp->timesVec(beta, yp->local_data(), alpha, xp->local_data_const());
    mDe->timesVec(0.,   yp->local_data(), alpha, xp->local_data_const()+mSp->n());
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


  virtual void addDiagonal(const hiopVector& d_)
  {
    assert(false && "not supported");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not supported");
  }
  virtual void addSubDiagonal(long long start, const hiopVector& d_)
  {
    assert(false && "not supported");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X)
  {
    const hiopMatrixMDS* pX=dynamic_cast<const hiopMatrixMDS*>(&X);
    if(pX==NULL) {
      assert(false && "operation only supported for hiopMatrixMDS left operand");
    }
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

  inline int sp_nnz() const { return mSp->nnz(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double** de_local_data() { return mDe->local_data(); }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
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


  virtual void addDiagonal(const hiopVector& d_)
  {
    assert(false && "not supported");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not supported");
  }
  virtual void addSubDiagonal(long long start, const hiopVector& d_)
  {
    assert(false && "not supported");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X)
  {
    const hiopMatrixSymBlockDiagMDS* pX=dynamic_cast<const hiopMatrixSymBlockDiagMDS*>(&X);
    if(pX==NULL) {
      assert(false && "operation only supported for hiopMatrixMDS left operand");
    }
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

  virtual inline long long m() const {return mSp->m();}
  virtual inline long long n() const {return mSp->n()+mDe->n();}
  inline long long n_sp() const {return mSp->n();}
  inline long long n_de() const {return  mDe->n();}

  inline int sp_nnz() const { return mSp->nnz(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double** de_local_data() { return mDe->local_data(); }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
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
