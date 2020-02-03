#include "hiopMatrixComplexSparseTriplet.hpp"

#include "blasdefs.hpp"

namespace hiop
{
  hiopMatrixComplexSparseTriplet::hiopMatrixComplexSparseTriplet(int rows, int cols, int nnz)
  {
    stM = new hiopMatrixSparseTripletStorage<int, std::complex<double> >(rows, cols, nnz);
  }
  hiopMatrixComplexSparseTriplet::~hiopMatrixComplexSparseTriplet()
  {
    delete stM;
  }
  
  hiopMatrix* hiopMatrixComplexSparseTriplet::alloc_clone() const
  {
    return new hiopMatrixComplexSparseTriplet(stM->m(), stM->n(), stM->numberOfNonzeros());
  }
  
  hiopMatrix* hiopMatrixComplexSparseTriplet::new_copy() const
  {
    assert(false);
    return NULL;
  }

  void hiopMatrixComplexSparseTriplet::setToZero()
  {
    auto* values = stM->M();
    for(int i=0; i<stM->numberOfNonzeros(); i++)
      values[i]=0.;
  }
  void hiopMatrixComplexSparseTriplet::setToConstant(double c)
  {
    auto* values = stM->M();
    for(int i=0; i<stM->numberOfNonzeros(); i++)
      values[i]=c;

  }
  void hiopMatrixComplexSparseTriplet::setToConstant(std::complex<double> c)
  {
    auto* values = stM->M();
    for(int i=0; i<stM->numberOfNonzeros(); i++)
      values[i]=c;
  }

  double hiopMatrixComplexSparseTriplet::max_abs_value()
  {
    char norm='M'; int one=1, nnz=stM->numberOfNonzeros();
    dcomplex* M = new dcomplex[nnz];
    for(int it=0; it<nnz; it++) {
      M[it].re = stM->M()[it].real();
      M[it].im = stM->M()[it].imag();
    }
      
    double maxv = ZLANGE(&norm, &one, &nnz, M, &one, NULL);

    delete[] M;
    return maxv;
  }
  
}//end namespace
