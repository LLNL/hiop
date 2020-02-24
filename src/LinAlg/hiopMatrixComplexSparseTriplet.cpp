#include "hiopMatrixComplexSparseTriplet.hpp"

#include "hiop_blasdefs.hpp"

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
    hiop::dcomplex* M = reinterpret_cast<dcomplex*>(stM->M());
    
    double maxv = ZLANGE(&norm, &one, &nnz, M, &one, NULL);
    return maxv;
  }

  void hiopMatrixComplexSparseTriplet::print(FILE* file, const char* msg/*=NULL*/, 
					     int maxRows/*=-1*/, int maxCols/*=-1*/, 
					     int rank/*=-1*/) const 
  {
    int myrank=0, numranks=1; //this is a local object => always print
    
    int max_elems = maxRows>=0 ? maxRows : stM->numberOfNonzeros();
    max_elems = std::min(max_elems, stM->numberOfNonzeros());

    if(file==NULL) file=stdout;
    
    if(myrank==rank || rank==-1) {
      
      if(NULL==msg) {
	if(numranks>1)
	  fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n", 
		  m(), n(), numberOfNonzeros(), max_elems, myrank);
	else
	  fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n", 
		  m(), n(), numberOfNonzeros(), max_elems);
      } else {
	fprintf(file, "%s ", msg);
      }    
      
    // output matlab indices and input format
    fprintf(file, "iRow=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", stM->irow[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "jCol=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", stM->jcol[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "v=[");
    for(int it=0; it<max_elems; it++)
      //fprintf(file, "%22.16e+%22.16ei; ", stM->values[it].real(), stM->values[it].imag());
      fprintf(file, "%.6g+%.6gi; ", stM->values[it].real(), stM->values[it].imag());
    fprintf(file, "];\n");
  }
}

}//end namespace
