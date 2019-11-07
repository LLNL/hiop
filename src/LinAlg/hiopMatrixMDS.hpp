#ifndef HIOP_SPARSE_MATRIX_MDS
#define HIOP_SPARSE_MATRIX_MDS

#include "hiopVector.hpp"
#include "hiopMatrix.hpp"
#include "hiopMatrixSparseTriplet.hpp"

#include <cassert>

namespace hiop
{

/** Mixed Sparse-Dense blocks matrix  - it is not distributed
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

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(long long start, const hiopVector& d_);

  virtual void addMatrix(double alpha, const hiopMatrix& X);
  virtual double max_abs_value();

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;

  virtual long long m() const {return mSp->m();}
  virtual long long n() const {return mSp->n()+mDe->n();}

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
#endif
private:
  hiopMatrixSparseTriplet* mSp;
  hiopMatrixDense* mDe;
private:
  hiopMatrixMDS() {};
  hiopMatrixMDS(const hiopMatrixMDS&) {};
};
}

#endif
